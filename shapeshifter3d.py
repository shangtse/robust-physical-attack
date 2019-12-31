#!/usr/bin/env python3.6
#
# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#
import os
import sys
import time
import math
import logging
import numpy as np
import tensorflow as tf

from functools import partial
from collections import OrderedDict

# <ordered>
from lucid.misc.gl.glcontext import create_opengl_context
from OpenGL import GL as gl
from lucid.misc.gl import glrenderer
# </ordered>

from lucid.misc.gl import meshutil
from lucid.optvis.param.spatial import sample_bilinear

from shapeshifter import parse_args
from shapeshifter import load_and_tile_images
from shapeshifter import create_model
from shapeshifter import create_textures
from shapeshifter import create_attack
from shapeshifter import create_evaluation
from shapeshifter import batch_accumulate
from shapeshifter import plot

def main():
    # Parse args
    args = parse_args(object_yaw_min=0, object_yaw_max=360,
                      object_pitch_min=-10, object_pitch_max=30,
                      object_z_min=15, object_z_max=60)

    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    log = logging.getLogger('shapeshifter')
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(levelname)s %(name)s: %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.propagate = False

    # Set seeds from the start
    if args.seed:
        log.debug("Setting seed")
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    # Load textures, textures_masks, backgrounds, and mesh
    log.debug("Loading textures, textures masks, backgrounds, and mesh")
    backgrounds = load_and_tile_images(args.backgrounds)

    textures = load_and_tile_images(args.textures)
    textures_masks = (load_and_tile_images(args.textures_masks)[:, :, :, :1] >= 0.5).astype(np.float32)
    assert(textures.shape[:3] == textures_masks.shape[:3])

    objects = [meshutil.normalize_mesh(meshutil.load_obj(obj)) for obj in args.objects]
    objects_masks = None

    # Create OpenGL context and mesh renderer
    log.debug("Creating renderer")
    create_opengl_context((backgrounds.shape[2], backgrounds.shape[1]))
    renderer = glrenderer.MeshRenderer((backgrounds.shape[2], backgrounds.shape[1]))

    # Create test data
    generate_data_partial = partial(generate_data,
                                    renderer=renderer,
                                    backgrounds=backgrounds,
                                    objects=objects,
                                    objects_masks=objects_masks,
                                    objects_class=args.target_class,
                                    objects_transforms={'yaw_range': args.object_yaw_range,
                                                        'yaw_bins': args.object_yaw_bins,
                                                        'yaw_fn': args.object_yaw_fn,

                                                        'pitch_range': args.object_pitch_range,
                                                        'pitch_bins': args.object_pitch_bins,
                                                        'pitch_fn': args.object_pitch_fn,

                                                        'roll_range': args.object_roll_range,
                                                        'roll_bins': args.object_roll_bins,
                                                        'roll_fn': args.object_roll_fn,

                                                        'x_range': args.object_x_range,
                                                        'x_bins': args.object_x_bins,
                                                        'x_fn': args.object_x_fn,

                                                        'y_range': args.object_y_range,
                                                        'y_bins': args.object_y_bins,
                                                        'y_fn': args.object_y_fn,

                                                        'z_range': args.object_z_range,
                                                        'z_bins': args.object_z_bins,
                                                        'z_fn': args.object_z_fn},
                                    textures_transforms={'yaw_range': args.texture_yaw_range,
                                                         'yaw_bins': args.texture_yaw_bins,
                                                         'yaw_fn': args.texture_yaw_fn,

                                                         'pitch_range': args.texture_pitch_range,
                                                         'pitch_bins': args.texture_pitch_bins,
                                                         'pitch_fn': args.texture_pitch_fn,

                                                         'roll_range': args.texture_roll_range,
                                                         'roll_bins': args.texture_roll_bins,
                                                         'roll_fn': args.texture_roll_fn,

                                                         'x_range': args.texture_x_range,
                                                         'x_bins': args.texture_x_bins,
                                                         'x_fn': args.texture_x_fn,

                                                         'y_range': args.texture_y_range,
                                                         'y_bins': args.texture_y_bins,
                                                         'y_fn': args.texture_y_fn,

                                                         'z_range': args.texture_z_range,
                                                         'z_bins': args.texture_z_bins,
                                                         'z_fn': args.texture_z_fn},
                                    seed=args.seed)

    # Create adversarial textures, render mesh using them, and pass rendered images into model. Finally, create summary statistics.
    log.debug("Creating perturbable texture")
    textures_var_, textures_ = create_textures(textures, textures_masks,
                                               use_spectral=args.spectral,
                                               soft_clipping=args.soft_clipping)

    log.debug("Creating rendered input images")
    input_images_ = create_rendered_images(args.batch_size, textures_)

    log.debug("Creating object detection model")
    predictions, detections, losses = create_model(input_images_, args.model_config, args.model_checkpoint, is_training=True)

    log.debug("Creating attack losses")
    victim_class_, target_class_, losses_summary_ = create_attack(textures_, textures_var_, predictions, losses,
                                                                  optimizer_name=args.optimizer, clip=args.sign_gradients)

    log.debug("Creating evaluation metrics")
    metrics_summary_, texture_summary_ = create_evaluation(victim_class_, target_class_,
                                                           textures_, textures_masks, input_images_, detections)

    summaries_ = tf.summary.merge([losses_summary_, metrics_summary_])

    global_init_op_ = tf.global_variables_initializer()
    local_init_op_ = tf.local_variables_initializer()

    # Create tensorboard file writer for train and test evaluations
    saver = tf.train.Saver([textures_var_, tf.train.get_global_step()])
    train_writer = None
    test_writer = None

    if args.logdir is not None:
        log.debug(f"Tensorboard logging: {args.logdir}")
        os.makedirs(args.logdir, exist_ok=True)

        arguments_summary_ = tf.summary.text('Arguments', tf.constant('```' + ' '.join(sys.argv[1:]) + '```'))
        # TODO: Save argparse

        graph = None
        if args.save_graph:
            log.debug("Graph will be saved to tensorboard")
            graph = tf.get_default_graph()

        train_writer = tf.summary.FileWriter(args.logdir + '/train', graph=graph)
        test_writer = tf.summary.FileWriter(args.logdir + '/test')

        # Find existing checkpoint
        os.makedirs(args.logdir + '/checkpoints', exist_ok=True)
        checkpoint_path = tf.train.latest_checkpoint(args.logdir + '/checkpoints')
        args.checkpoint = checkpoint_path

    # Create session
    log.debug("Creating session")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.run(global_init_op_)
    sess.run(local_init_op_)

    # Set initial texture
    if args.checkpoint is not None:
        log.debug(f"Restoring from checkpoint: {args.checkpoint}")
        saver.restore(sess, args.checkpoint)
    else:
        if args.gray_start:
            log.debug("Setting texture to gray")
            textures = np.zeros_like(textures) + 128/255

        if args.random_start > 0:
            log.debug(f"Adding uniform random perturbation texture with at most {args.random_start}/255 per pixel")
            textures = textures + np.random.randint(size=textures.shape, low=-args.random_start, high=args.random_start)/255

        sess.run('project_op', { 'textures:0': textures  })

    # Get global step
    step = sess.run('global_step:0')

    if train_writer is not None:
        log.debug("Running arguments summary")
        summary = sess.run(arguments_summary_)

        train_writer.add_summary(summary, step)
        test_writer.add_summary(summary, step)

    loss_tensors = ['total_loss:0',
                    'total_rpn_cls_loss:0', 'total_rpn_loc_loss:0',
                    'total_rpn_foreground_loss:0', 'total_rpn_background_loss:0',
                    'total_rpn_cw_loss:0',
                    'total_box_cls_loss:0', 'total_box_loc_loss:0',
                    'total_box_target_loss:0', 'total_box_victim_loss:0',
                    'total_box_target_cw_loss:0', 'total_box_victim_cw_loss:0',
                    'total_sim_loss:0',
                    'grad_l2:0', 'grad_linf:0']

    metric_tensors = ['proposal_average_precision:0', 'victim_average_precision:0', 'target_average_precision:0']

    output_tensors = loss_tensors + metric_tensors

    log.info('global_step [%s]', ', '.join([tensor.replace(':0', '').replace('total_', '') for tensor in output_tensors]))

    test_feed_dict = { 'learning_rate:0': args.learning_rate,
                       'momentum:0': args.momentum,
                       'decay:0': args.decay,

                       'rpn_iou_thresh:0': args.rpn_iou_threshold,
                       'rpn_cls_weight:0': args.rpn_cls_weight,
                       'rpn_loc_weight:0': args.rpn_loc_weight,
                       'rpn_foreground_weight:0': args.rpn_foreground_weight,
                       'rpn_background_weight:0': args.rpn_background_weight,
                       'rpn_cw_weight:0': args.rpn_cw_weight,
                       'rpn_cw_conf:0': args.rpn_cw_conf,

                       'box_iou_thresh:0': args.box_iou_threshold,
                       'box_cls_weight:0': args.box_cls_weight,
                       'box_loc_weight:0': args.box_loc_weight,
                       'box_target_weight:0': args.box_target_weight,
                       'box_victim_weight:0': args.box_victim_weight,
                       'box_target_cw_weight:0': args.box_target_cw_weight,
                       'box_target_cw_conf:0': args.box_target_cw_conf,
                       'box_victim_cw_weight:0': args.box_victim_cw_weight,
                       'box_victim_cw_conf:0': args.box_victim_cw_conf,

                       'sim_weight:0': args.sim_weight,

                       'victim_class:0': args.victim_class,
                       'target_class:0': args.target_class }

    # Keep attacking until CTRL+C. The only issue is that we may be in the middle of some operation.
    try:
        log.debug("Entering attacking loop (use ctrl+c to exit)")
        while True:
            # Run summaries as necessary
            if args.logdir and step % args.save_checkpoint_every == 0:
                log.debug("Saving checkpoint")
                saver.save(sess, args.logdir + '/checkpoints/texture', global_step=step, write_meta_graph=False, write_state=True)

            if step % args.save_texture_every == 0 and test_writer is not None:
                log.debug("Writing texture summary")
                test_texture = sess.run(texture_summary_)
                test_writer.add_summary(test_texture, step)

            if step % args.save_test_every == 0:
                log.debug("Runnning test summaries")
                start_time = time.time()

                sess.run(local_init_op_)
                batch_accumulate(sess, test_feed_dict,
                                 args.test_batch_size, args.batch_size,
                                 generate_data_partial,
                                 detections, predictions, args.categories)

                end_time = time.time()
                log.debug(f"Loss accumulation took {end_time - start_time} seconds")

                test_output = sess.run(output_tensors, test_feed_dict)
                log.info('test %d %s', step, test_output)

                if test_writer is not None:
                    log.debug("Writing test summaries")
                    test_summaries = sess.run(summaries_, test_feed_dict)
                    test_writer.add_summary(test_summaries, step)

            # Create train feed_dict
            train_feed_dict = test_feed_dict.copy()

            train_feed_dict['image_channel_multiplicative_noise:0'] = args.image_multiplicative_channel_noise_range
            train_feed_dict['image_channel_additive_noise:0'] = args.image_additive_channel_noise_range
            train_feed_dict['image_pixel_multiplicative_noise:0'] = args.image_multiplicative_pixel_noise_range
            train_feed_dict['image_pixel_additive_noise:0'] = args.image_additive_pixel_noise_range
            train_feed_dict['image_gaussian_noise_stddev:0'] = args.image_gaussian_noise_stddev_range

            train_feed_dict['texture_channel_multiplicative_noise:0'] = args.texture_multiplicative_channel_noise_range
            train_feed_dict['texture_channel_additive_noise:0'] = args.texture_additive_channel_noise_range
            train_feed_dict['texture_pixel_multiplicative_noise:0'] = args.texture_multiplicative_pixel_noise_range
            train_feed_dict['texture_pixel_additive_noise:0'] = args.texture_additive_pixel_noise_range
            train_feed_dict['texture_gaussian_noise_stddev:0'] = args.texture_gaussian_noise_stddev_range

            # Zero out gradient accumulation, losses, and metrics, then accumulate batches
            log.debug("Starting gradient accumulation...")
            start_time = time.time()

            sess.run(local_init_op_)
            batch_accumulate(sess, train_feed_dict,
                             args.train_batch_size, args.batch_size,
                             generate_data_partial,
                             detections, predictions, args.categories)

            end_time = time.time()
            log.debug(f"Gradient accumulation took {end_time - start_time} seconds")

            train_output = sess.run(output_tensors, train_feed_dict)
            log.info('train %d %s', step, train_output)

            if step % args.save_train_every == 0 and train_writer is not None:
                log.debug("Writing train summaries")
                train_summaries = sess.run(summaries_, test_feed_dict)
                train_writer.add_summary(train_summaries, step)

            # Update textures and project texture to feasible set
            # TODO: We can probably run these together but probably need some control dependency
            log.debug("Projecting attack")
            sess.run('attack_op', train_feed_dict)
            sess.run('project_op')
            step = sess.run('global_step:0')

    except KeyboardInterrupt:
        log.warn('Interrupted')

    finally:
        if test_writer is not None:
            test_writer.close()

        if train_writer is not None:
            train_writer.close()

        if sess is not None:
            sess.close()

def create_rendered_images(batch_size, textures):
    backgrounds_ = tf.placeholder(tf.float32, [batch_size, None, None, 3], name='backgrounds')
    frames_ = tf.placeholder(tf.float32, [batch_size, None, None, 4], name='frames')

    texture_channel_multiplicative_noise_ = tf.placeholder_with_default([1., 1.], [2], name='texture_channel_multiplicative_noise')
    texture_channel_additive_noise_ = tf.placeholder_with_default([0., 0.], [2], name='texture_channel_additive_noise')

    texture_pixel_multiplicative_noise_ = tf.placeholder_with_default([1., 1.], [2], name='texture_pixel_multiplicative_noise')
    texture_pixel_additive_noise_ = tf.placeholder_with_default([0., 0.], [2], name='texture_pixel_additive_noise')

    texture_gaussian_noise_stddev_ = tf.placeholder_with_default([0., 0.], [2], name='texture_gaussian_noise_stddev')

    image_channel_multiplicative_noise_ = tf.placeholder_with_default([1., 1.], [2], name='image_channel_multiplicative_noise')
    image_channel_additive_noise_ = tf.placeholder_with_default([0., 0.], [2], name='image_channel_additive_noise')

    image_pixel_multiplicative_noise_ = tf.placeholder_with_default([1., 1.], [2], name='image_pixel_multiplicative_noise')
    image_pixel_additive_noise_ = tf.placeholder_with_default([0., 0.], [2], name='image_pixel_additive_noise')

    image_gaussian_noise_stddev_ = tf.placeholder_with_default([0., 0.], [2], name='image_gaussian_noise_stddev')

    IDENTITY_KERNEL = [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]
    image_gaussian_blur_kernel_ = tf.placeholder_with_default(IDENTITY_KERNEL, [None, None], name='image_gaussian_blur_kernel')
    image_gaussian_blur_kernel_ = image_gaussian_blur_kernel_[:, :, tf.newaxis, tf.newaxis]
    image_gaussian_blur_kernel_ = tf.tile(image_gaussian_blur_kernel_, [1, 1, 3, 1])

    # TODO: This could probably be made faster by removing random elements to outside of loop
    def render_frame(frame_):
        textures_ = textures

        # Add noise to textures
        textures_ = textures_ * tf.random_uniform([3], texture_channel_multiplicative_noise_[0], texture_channel_multiplicative_noise_[1])
        textures_ = textures_ + tf.random_uniform([3], texture_channel_additive_noise_[0], texture_channel_additive_noise_[1])

        textures_ = textures_ * tf.random_uniform([], texture_pixel_multiplicative_noise_[0], texture_pixel_multiplicative_noise_[1])
        textures_ = textures_ + tf.random_uniform([], texture_pixel_additive_noise_[0], texture_pixel_additive_noise_[1])

        textures_ = textures_ + tf.random_normal(textures_.shape, stddev=tf.random_uniform([], texture_gaussian_noise_stddev_[0], texture_gaussian_noise_stddev_[1]))

        #textures_ = tf.clip_by_value(textures_, 0.0, 1.0)

        # Render
        uvf_ = frame_[..., :3]
        image_ = sample_bilinear(textures_, uvf_)

        # Composite onto background
        # FIXME: This only really works with batch_size=1
        alpha_ = frame_[..., 3:]
        image_ = image_*alpha_ + backgrounds_[0]*(1 - alpha_)

        # Blur image
        image_ = image_[tf.newaxis, :, :, :]
        image_ = tf.nn.depthwise_conv2d(image_, image_gaussian_blur_kernel_, strides=[1, 1, 1, 1], padding='SAME')
        image_ = image_[0]

        # Blur alpha
        alpha_ = alpha_[tf.newaxis, :, :, :]
        alpha_ = tf.nn.depthwise_conv2d(alpha_, image_gaussian_blur_kernel_[:, :, :1, :], strides=[1, 1, 1, 1], padding='SAME')
        alpha_ = alpha_[0]

        # Recomposite blurred image onto background
        # FIXME: This only really works with batch_size=1
        image_ = image_*alpha_ + backgrounds_[0]*(1 - alpha_)

        # Add noise to image
        image_ = image_ * tf.random_uniform([3], image_channel_multiplicative_noise_[0], image_channel_multiplicative_noise_[1])
        image_ = image_ + tf.random_uniform([3], image_channel_additive_noise_[0], image_channel_additive_noise_[1])

        image_ = image_ * tf.random_uniform([], image_pixel_multiplicative_noise_[0], image_pixel_multiplicative_noise_[1])
        image_ = image_ + tf.random_uniform([], image_pixel_additive_noise_[0], image_pixel_additive_noise_[1])

        image_ = image_ + tf.random_normal(tf.shape(image_), stddev=tf.random_uniform([], image_gaussian_noise_stddev_[0], image_gaussian_noise_stddev_[1]))

        #image_ = tf.clip_by_value(image_, 0.0, 1.0)

        return image_

    input_images_ = tf.map_fn(render_frame, frames_, dtype=(tf.float32))
    # TODO: Can we move image compositing to out of render_frame?
    # TODO: Move noising of image to out of render_frame to here

    input_images_ = tf.fake_quant_with_min_max_args(input_images_, min=0., max=1., num_bits=8)
    input_images_ = tf.identity(input_images_, name='input_images')

    return input_images_

def generate_data(count, renderer, backgrounds, objects, objects_masks, objects_class, objects_transforms, textures_transforms, seed=None):
    if seed:
        np.random.seed(seed)

    origin = (0, 0)
    objects_transforms = generate_transforms(origin, count, **objects_transforms)
    frames, bboxes = generate_frames(renderer, objects, objects_transforms)
    backgrounds = backgrounds[np.random.choice(backgrounds.shape[0], size=count)]

    data = {'frames:0': frames,
            'backgrounds:0': backgrounds,
            'groundtruth_boxes_%d:0': bboxes,
            'groundtruth_classes_%d:0': np.full(bboxes.shape[0:2], objects_class - 1),
            'groundtruth_weights_%d:0': np.ones(bboxes.shape[0:2])}

    return data

def generate_transforms(origin, count,
                        yaws=None, yaw_range=(0, 0), yaw_bins=100, yaw_fn=np.linspace,
                        pitchs=None, pitch_range=(0, 0), pitch_bins=100, pitch_fn=np.linspace,
                        rolls=None, roll_range=(0, 0), roll_bins=100, roll_fn=np.linspace,
                        xs=None, x_range=(0, 0), x_bins=100, x_fn=np.linspace,
                        ys=None, y_range=(0, 0), y_bins=100, y_fn=np.linspace,
                        zs=None, z_range=(0, 0), z_bins=100, z_fn=np.linspace):
    # Discretize ranges
    if yaws is None:
        yaws = yaw_fn(*yaw_range, num=yaw_bins)
    if pitchs is None:
        pitchs = pitch_fn(*pitch_range, num=pitch_bins)
    if rolls is None:
        rolls = roll_fn(*roll_range, num=roll_bins)
    if xs is None:
        xs = x_fn(*x_range, num=x_bins)
    if ys is None:
        ys = y_fn(*y_range, num=y_bins)
    if zs is None:
        zs = z_fn(*z_range, num=z_bins)

    # Convert from degrees to radians
    yaws = yaws * np.pi/180
    pitchs = pitchs * np.pi/180
    rolls = rolls * np.pi/180

    # Choose randomly
    yaws = np.random.choice(yaws, count)
    pitchs = np.random.choice(pitchs, count)
    rolls = np.random.choice(rolls, count)
    xs = np.random.choice(xs, count)
    ys = np.random.choice(ys, count)
    zs = np.random.choice(zs, count)

    # Get transforms for each options
    transforms = []

    for yaw, pitch, roll, x, y, z in zip(yaws, pitchs, rolls, xs, ys, zs):
        transform = meshutil.lookat([z*np.cos(pitch)*np.sin(yaw), z*np.sin(pitch), z*np.cos(pitch)*np.cos(yaw)], [x, y, 0])
        transforms.append(transform)

    return transforms

def generate_frames(renderer, meshes, modelviews):
    frames = []
    for modelview in modelviews:
        mesh = np.random.choice(meshes)
        frame = renderer.render_mesh(mesh['position'], mesh['uv'], mesh['faces'], modelview=modelview)

        # 2 here refers the the texture index, so we multiply by the number of faces to get integers that we can use to index
        # since OpenGL only likes numbers between 0 and 1.
        frame[:, :, 2] = np.round(frame[:, :, 2] * (len(mesh['faces']) - 1))

        frames.append(frame)
    frames = np.array(frames)

    # Extract bounding boxes from 3rd frame data
    rows = np.any(frames[:, :, :, 3] > 0., axis=1)
    cols = np.any(frames[:, :, :, 3] > 0., axis=2)

    ymin = np.argmax(cols, axis=1)
    ymax = frames.shape[1] - np.argmax(cols[:, ::-1], axis=1) - 1
    xmin = np.argmax(rows, axis=1)
    xmax = frames.shape[2] - np.argmax(rows[:, ::-1], axis=1) - 1

    bboxes = np.array([[ymin / frames.shape[1]], [xmin / frames.shape[2]], [ymax / frames.shape[1]], [xmax / frames.shape[2]]]).T

    return frames, bboxes

if __name__ == '__main__':
    main()
