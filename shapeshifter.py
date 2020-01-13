#
# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#
import math
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import variables_helper
from object_detection.utils import visualization_utils

from object_detection.builders import model_builder

from object_detection.utils.object_detection_evaluation import ObjectDetectionEvaluator
from object_detection.core.standard_fields import InputDataFields, DetectionResultFields

def parse_args(texture_yaw_min=0, texture_yaw_max=0,
               texture_pitch_min=0, texture_pitch_max=0,
               texture_roll_min=0, texture_roll_max=0,
               texture_x_min=0, texture_x_max=0,
               texture_y_min=0, texture_y_max=0,
               texture_z_min=0, texture_z_max=0,
               object_yaw_min=0, object_yaw_max=0,
               object_pitch_min=0, object_pitch_max=0,
               object_roll_min=0, object_roll_max=0,
               object_x_min=0, object_x_max=0,
               object_y_min=0, object_y_max=0,
               object_z_min=0, object_z_max=0):
    parser = argparse.ArgumentParser()
    parser.register('type', 'strbool', lambda x: x.lower() in ('true', '1', 'yes'))

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=None)

    group = parser.add_argument_group('Object Detection Model')
    group.add_argument('--model', type=str, required=True)
    group.add_argument('--batch-size', type=int, default=1)
    group.add_argument('--train-batch-size', type=int, default=1000)
    group.add_argument('--test-batch-size', type=int, default=1000)

    group = parser.add_argument_group('Inputs')
    group.add_argument('--backgrounds', type=str, nargs='+', required=True)

    group.add_argument('--textures', type=str, nargs='+', required=True)
    group.add_argument('--textures-masks', type=str, nargs='+', required=True)

    group.add_argument('--texture-yaw-min', type=float, default=texture_yaw_min)
    group.add_argument('--texture-yaw-max', type=float, default=texture_yaw_max)
    group.add_argument('--texture-yaw-bins', type=int, default=100)
    group.add_argument('--texture-yaw-logspace', dest='texture_yaw_fn', action='store_const', const=np.logspace, default=np.linspace)

    group.add_argument('--texture-pitch-min', type=float, default=texture_pitch_min)
    group.add_argument('--texture-pitch-max', type=float, default=texture_pitch_max)
    group.add_argument('--texture-pitch-bins', type=int, default=100)
    group.add_argument('--texture-pitch-logspace', dest='texture_pitch_fn', action='store_const', const=np.logspace, default=np.linspace)

    group.add_argument('--texture-roll-min', type=float, default=texture_roll_min)
    group.add_argument('--texture-roll-max', type=float, default=texture_roll_max)
    group.add_argument('--texture-roll-bins', type=int, default=100)
    group.add_argument('--texture-roll-logspace', dest='texture_roll_fn', action='store_const', const=np.logspace, default=np.linspace)

    group.add_argument('--texture-x-min', type=float, default=texture_x_min)
    group.add_argument('--texture-x-max', type=float, default=texture_x_max)
    group.add_argument('--texture-x-bins', type=int, default=100)
    group.add_argument('--texture-x-logspace', dest='texture_x_fn', action='store_const', const=np.logspace, default=np.linspace)

    group.add_argument('--texture-y-min', type=float, default=texture_y_min)
    group.add_argument('--texture-y-max', type=float, default=texture_y_max)
    group.add_argument('--texture-y-bins', type=int, default=100)
    group.add_argument('--texture-y-logspace', dest='texture_y_fn', action='store_const', const=np.logspace, default=np.linspace)

    group.add_argument('--texture-z-min', type=float, default=texture_z_min)
    group.add_argument('--texture-z-max', type=float, default=texture_z_max)
    group.add_argument('--texture-z-bins', type=int, default=100)
    group.add_argument('--texture-z-logspace', dest='texture_z_fn', action='store_const', const=np.logspace, default=np.linspace)

    group.add_argument('--texture-multiplicative-channel-noise-min', type=float, default=1.)
    group.add_argument('--texture-multiplicative-channel-noise-max', type=float, default=1.)
    group.add_argument('--texture-additive-channel-noise-min', type=float, default=0.)
    group.add_argument('--texture-additive-channel-noise-max', type=float, default=0.)
    group.add_argument('--texture-multiplicative-pixel-noise-min', type=float, default=1.)
    group.add_argument('--texture-multiplicative-pixel-noise-max', type=float, default=2.)
    group.add_argument('--texture-additive-pixel-noise-min', type=float, default=0.)
    group.add_argument('--texture-additive-pixel-noise-max', type=float, default=0.)
    group.add_argument('--texture-gaussian-noise-stddev-min', type=float, default=0.)
    group.add_argument('--texture-gaussian-noise-stddev-max', type=float, default=0.)

    group.add_argument('--objects', type=str, nargs='+', required=True)

    group.add_argument('--object-yaw-min', type=float, default=object_yaw_min)
    group.add_argument('--object-yaw-max', type=float, default=object_yaw_max)
    group.add_argument('--object-yaw-bins', type=int, default=100)
    group.add_argument('--object-yaw-logspace', dest='object_yaw_fn', action='store_const', const=np.logspace, default=np.linspace)

    group.add_argument('--object-pitch-min', type=float, default=object_pitch_min)
    group.add_argument('--object-pitch-max', type=float, default=object_pitch_max)
    group.add_argument('--object-pitch-bins', type=int, default=100)
    group.add_argument('--object-pitch-logspace', dest='object_pitch_fn', action='store_const', const=np.logspace, default=np.linspace)

    group.add_argument('--object-roll-min', type=float, default=object_roll_min)
    group.add_argument('--object-roll-max', type=float, default=object_roll_max)
    group.add_argument('--object-roll-bins', type=int, default=100)
    group.add_argument('--object-roll-logspace', dest='object_roll_fn', action='store_const', const=np.logspace, default=np.linspace)

    group.add_argument('--object-x-min', type=float, default=object_x_min)
    group.add_argument('--object-x-max', type=float, default=object_x_max)
    group.add_argument('--object-x-bins', type=int, default=100)
    group.add_argument('--object-x-logspace', dest='object_x_fn', action='store_const', const=np.logspace, default=np.linspace)

    group.add_argument('--object-y-min', type=float, default=object_y_min)
    group.add_argument('--object-y-max', type=float, default=object_y_max)
    group.add_argument('--object-y-bins', type=int, default=100)
    group.add_argument('--object-y-logspace', dest='object_y_fn', action='store_const', const=np.logspace, default=np.linspace)

    group.add_argument('--object-z-min', type=float, default=object_z_min)
    group.add_argument('--object-z-max', type=float, default=object_z_max)
    group.add_argument('--object-z-bins', type=int, default=100)
    group.add_argument('--object-z-logspace', dest='object_z_fn', action='store_const', const=np.logspace, default=np.linspace)

    group.add_argument('--object-multiplicative-channel-noise-min', type=float, default=1.)
    group.add_argument('--object-multiplicative-channel-noise-max', type=float, default=1.)
    group.add_argument('--object-additive-channel-noise-min', type=float, default=0.)
    group.add_argument('--object-additive-channel-noise-max', type=float, default=0.)
    group.add_argument('--object-multiplicative-pixel-noise-min', type=float, default=1.)
    group.add_argument('--object-multiplicative-pixel-noise-max', type=float, default=1.)
    group.add_argument('--object-additive-pixel-noise-min', type=float, default=0.)
    group.add_argument('--object-additive-pixel-noise-max', type=float, default=0.)
    group.add_argument('--object-gaussian-noise-stddev-min', type=float, default=0.)
    group.add_argument('--object-gaussian-noise-stddev-max', type=float, default=0.)

    group.add_argument('--image-multiplicative-channel-noise-min', type=float, default=1.)
    group.add_argument('--image-multiplicative-channel-noise-max', type=float, default=1.)
    group.add_argument('--image-additive-channel-noise-min', type=float, default=0.)
    group.add_argument('--image-additive-channel-noise-max', type=float, default=0.)
    group.add_argument('--image-multiplicative-pixel-noise-min', type=float, default=1.)
    group.add_argument('--image-multiplicative-pixel-noise-max', type=float, default=1.)
    group.add_argument('--image-additive-pixel-noise-min', type=float, default=0.)
    group.add_argument('--image-additive-pixel-noise-max', type=float, default=0.)
    group.add_argument('--image-gaussian-noise-stddev-min', type=float, default=0.)
    group.add_argument('--image-gaussian-noise-stddev-max', type=float, default=0.)

    group = parser.add_argument_group('Attack')
    group.add_argument('--optimizer', type=str, choices=['gd', 'momentum', 'rmsprop', 'adam'], default='gd')
    group.add_argument('--learning-rate', type=float, default=1.0)
    group.add_argument('--momentum', type=float,  default=0.)
    group.add_argument('--decay', type=float, default=0.)
    group.add_argument('--sign-gradients', type='strbool', default=False)

    group.add_argument('--gray-start', action='store_true')
    group.add_argument('--random-start', type=int, default=0)

    group.add_argument('--spectral', type='strbool', default=True)
    group.add_argument('--soft-clipping', type='strbool', default=False)

    group.add_argument('--target', type=str, default='bird')
    group.add_argument('--victim', type=str, default='person')

    group.add_argument('--rpn-iou-threshold', type=float, default=0.7)
    group.add_argument('--rpn-cls-weight', type=float, default=1.)
    group.add_argument('--rpn-loc-weight', type=float, default=2.)
    group.add_argument('--rpn-foreground-weight', type=float, default=0.)
    group.add_argument('--rpn-background-weight', type=float, default=0.)
    group.add_argument('--rpn-cw-weight', type=float, default=0.)
    group.add_argument('--rpn-cw-conf', type=float, default=0.)

    group.add_argument('--box-iou-threshold', type=float, default=0.5)
    group.add_argument('--box-cls-weight', type=float, default=1.)
    group.add_argument('--box-loc-weight', type=float, default=2.)
    group.add_argument('--box-target-weight', type=float, default=0.)
    group.add_argument('--box-victim-weight', type=float, default=0.)
    group.add_argument('--box-target-cw-weight', type=float, default=0.)
    group.add_argument('--box-target-cw-conf', type=float, default=0.)
    group.add_argument('--box-victim-cw-weight', type=float, default=0.)
    group.add_argument('--box-victim-cw-conf', type=float, default=0.)

    group.add_argument('--sim-weight', type=float, default=0.)

    group = parser.add_argument_group('Metrics')
    group.add_argument('--logdir', type=str)
    group.add_argument('--save-graph', action='store_true')
    group.add_argument('--save-train-every', type=int, default=1)
    group.add_argument('--save-texture-every', type=int, default=10)
    group.add_argument('--save-checkpoint-every', type=int, default=10)
    group.add_argument('--save-test-every', type=int, default=10)

    args = parser.parse_args()

    # Create tuples from -min -max args
    for key in list(vars(args)):
        if key.endswith('_min'):
            key = key[:-4]
            setattr(args, key + '_range', (getattr(args, key + '_min'), getattr(args, key + '_max')))

    # Parse args.model directory
    args.model_config = args.model + '/pipeline.config'
    args.model_checkpoint = args.model + '/model.ckpt'
    args.model_labels = args.model + '/label_map.pbtxt'

    # Read labels and set target class and victim class ids
    args.label_map = label_map_util.get_label_map_dict(args.model_labels, use_display_name=True)
    args.category_index = label_map_util.create_category_index_from_labelmap(args.model_labels)
    args.categories = list(args.category_index.values())

    args.target_class = args.label_map[args.target]
    args.victim_class = args.label_map[args.victim]

    return args

def load_and_tile_images(paths, replicates=0, resize=None):
    images = [Image.open(path) for path in paths]
    if resize is not None:
        images = [image.resize(resize) for image in images]
    images = [np.tile(image, (1+replicates, 1, 1, 1)) for image in images]
    images = np.concatenate(images, axis=0)
    images = (images / 255.0).astype(np.float32)

    return images

def create_textures(initial_textures, textures_masks, use_spectral=True, soft_clipping=False):
    N, H, W, C = initial_textures.shape

    initial_textures_ = tf.constant(initial_textures, name='initial_textures')

    # Create texture variables
    if use_spectral:
        textures_var_ = tf.Variable(np.zeros((2, N, C, H, W//2 + 1 + (W % 2))), dtype=tf.float32, name='textures_var')
        textures_ = textures_var_
        textures_ = tf.complex(textures_[0], textures_[1])
        textures_ = tf.map_fn(tf.spectral.irfft2d, textures_, dtype=tf.float32)
        textures_ = tf.transpose(textures_, (0, 2, 3, 1))
    else:
        textures_var_ = tf.Variable(np.zeros((N, H, W, C)), dtype=tf.float32, name='textures_var')
        textures_ = textures_var_

    textures_ = tf.identity(textures_, name='textures')

    # Invert textures for projection step
    projected_textures_ = initial_textures_ * (1.0 - textures_masks) + textures_ * textures_masks

    if soft_clipping:
        projected_textures_ = tf.nn.sigmoid(projected_textures_)
    else:
        projected_textures_ = tf.clip_by_value(projected_textures_, 0., 1.)

    if use_spectral:
        projected_textures_ = tf.transpose(projected_textures_, (0, 3, 1, 2))
        projected_textures_ = tf.map_fn(tf.spectral.rfft2d, projected_textures_, dtype=tf.complex64)
        projected_textures_ = tf.stack([tf.real(projected_textures_), tf.imag(projected_textures_)])

    projected_textures_ = tf.identity(projected_textures_, name='projected_textures')

    project_op_ = tf.assign(textures_var_, projected_textures_, name='project_op')

    return textures_var_, textures_

def create_model(input_images, pipeline_config_path, fine_tune_checkpoint, is_training=False):
    gt_boxes = [tf.placeholder(tf.float32, (None, 4), name=f"groundtruth_boxes_{i}") for i in range(input_images.shape[0])]
    gt_classes = [tf.placeholder(tf.int32, (None,), name=f"groundtruth_classes_{i}") for i in range(input_images.shape[0])]
    gt_weights = [tf.placeholder(tf.float32, (None,), name=f"groundtruth_weights_{i}") for i in range(input_images.shape[0])]

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)

    # Ensure second_stage_batch_size is equal to first_stage_max_proposals
    # This makes detection deterministic since we're not sampling proposals.
    # One other change to keep in mind is that when is_training=True, anchors are pruned
    # that venture outside the image window, while when is_training=False, achors are clipped
    # to the image window. We turn this off by applying the patch in faster_rcnn_meta_arch.patch
    configs['model'].faster_rcnn.second_stage_batch_size = configs['model'].faster_rcnn.first_stage_max_proposals

    # Create model with ground truth
    detection_model = model_builder.build(model_config=configs['model'], is_training=is_training, add_summaries=False)

    detection_model.provide_groundtruth(groundtruth_boxes_list=gt_boxes,
                                        groundtruth_classes_list=[tf.one_hot(gt_class_, detection_model.num_classes) for gt_class_ in gt_classes],
                                        groundtruth_weights_list=gt_weights)

    # Create model pipeline
    preprocessed_images_, true_image_shapes_  = detection_model.preprocess(255.0*input_images)
    predictions = detection_model.predict(preprocessed_images_, true_image_shapes_)
    detections = detection_model.postprocess(predictions, true_image_shapes_)
    losses = detection_model.loss(predictions, true_image_shapes_)

    # Set up variables to be initialize from checkpoint
    asg_map = detection_model.restore_map(fine_tune_checkpoint_type='detection', load_all_detection_checkpoint_vars=True)
    available_var_map = variables_helper.get_variables_available_in_checkpoint(asg_map, fine_tune_checkpoint, include_global_step=False)

    tf.train.init_from_checkpoint(fine_tune_checkpoint, available_var_map)

    return predictions, detections, losses

def create_attack(textures, textures_var, predictions, losses, optimizer_name='gd', clip=False):
    # Box Classsifier Loss
    box_cls_weight_ = tf.placeholder_with_default(1.0, [], name='box_cls_weight')
    box_cls_loss_ = losses['Loss/BoxClassifierLoss/classification_loss']
    weighted_box_cls_loss_ = tf.multiply(box_cls_loss_, box_cls_weight_, name='box_cls_loss')

    # Box Localizer Loss
    box_loc_weight_ = tf.placeholder_with_default(2.0, [], name='box_loc_weight')
    box_loc_loss_ = losses['Loss/BoxClassifierLoss/localization_loss']
    weighted_box_loc_loss_ = tf.multiply(box_loc_loss_, box_loc_weight_, name='box_loc_loss')

    # RPN Classifier Loss
    rpn_cls_weight_ = tf.placeholder_with_default(1.0, [], name='rpn_cls_weight')
    rpn_cls_loss_ = losses['Loss/RPNLoss/objectness_loss']
    weighted_rpn_cls_loss_ = tf.multiply(rpn_cls_loss_, rpn_cls_weight_, name='rpn_cls_loss')

    # RPN Localizer Loss
    rpn_loc_weight_ = tf.placeholder_with_default(2.0, [], name='rpn_loc_weight')
    rpn_loc_loss_ = losses['Loss/RPNLoss/localization_loss']
    weighted_rpn_loc_loss_ = tf.multiply(rpn_loc_loss_, rpn_loc_weight_, name='rpn_loc_loss')

    # Box Losses
    target_class_ = tf.placeholder(tf.int32, [], name='target_class')
    victim_class_ = tf.placeholder(tf.int32, [], name='victim_class')
    box_iou_thresh_ = tf.placeholder_with_default(0.5, [], name='box_iou_thresh')

    box_logits_ = predictions['class_predictions_with_background']
    box_logits_ = box_logits_[:, 1:] # Ignore background class
    victim_one_hot_ = tf.one_hot([victim_class_ - 1], box_logits_.shape[-1])
    target_one_hot_ = tf.one_hot([target_class_ - 1], box_logits_.shape[-1])
    box_iou_ = tf.get_default_graph().get_tensor_by_name('Loss/BoxClassifierLoss/Compare/IOU/Select:0')
    box_iou_ = tf.reshape(box_iou_, (-1,))
    box_targets_ = tf.cast(box_iou_ >= box_iou_thresh_, tf.float32)

    # Box Victim Loss
    box_victim_weight_ = tf.placeholder_with_default(0.0, [], name='box_victim_weight')

    box_victim_logits_ = box_logits_[:, victim_class_ - 1]
    box_victim_loss_ = box_victim_logits_*box_targets_
    box_victim_loss_ = tf.reduce_sum(box_victim_loss_)
    weighted_box_victim_loss_ = tf.multiply(box_victim_loss_, box_victim_weight_, name='box_victim_loss')

    # Box Target Loss
    box_target_weight_ = tf.placeholder_with_default(0.0, [], name='box_target_weight')

    box_target_logits_ = box_logits_[:, target_class_ - 1]
    box_target_loss_ = box_target_logits_*box_targets_
    box_target_loss_ = -1*tf.reduce_sum(box_target_loss_) # Maximize!
    weighted_box_target_loss_ = tf.multiply(box_target_loss_, box_target_weight_, name='box_target_loss')

    # Box Victim CW loss (untargeted, victim -> nonvictim)
    box_victim_cw_weight_ = tf.placeholder_with_default(0.0, [], name='box_victim_cw_weight')
    box_victim_cw_conf_ = tf.placeholder_with_default(0.0, [], name='box_victim_cw_conf')

    box_nonvictim_logits_ = tf.reduce_max(box_logits_ * (1 - victim_one_hot_) - 10000*victim_one_hot_, axis=-1)
    box_victim_cw_loss_ = tf.nn.relu(box_victim_logits_ - box_nonvictim_logits_ + box_victim_cw_conf_)
    box_victim_cw_loss_ = box_victim_cw_loss_*box_targets_
    box_victim_cw_loss_ = tf.reduce_sum(box_victim_cw_loss_)
    weighted_box_victim_cw_loss_ = tf.multiply(box_victim_cw_loss_, box_victim_cw_weight_, name='box_victim_cw_loss')

    # Box Target CW loss (targeted, nontarget -> target)
    box_target_cw_weight_ = tf.placeholder_with_default(0.0, [], name='box_target_cw_weight')
    box_target_cw_conf_ = tf.placeholder_with_default(0.0, [], name='box_target_cw_conf')

    box_nontarget_logits_ = tf.reduce_max(box_logits_ * (1 - target_one_hot_) - 10000*target_one_hot_, axis=-1)
    box_target_cw_loss_ = tf.nn.relu(box_nontarget_logits_ - box_target_logits_ + box_target_cw_conf_)
    box_target_cw_loss_ = box_target_cw_loss_*box_targets_
    box_target_cw_loss_ = tf.reduce_sum(box_target_cw_loss_)
    weighted_box_target_cw_loss_ = tf.multiply(box_target_cw_loss_, box_target_cw_weight_, name='box_target_cw_loss')

    # RPN Losses
    rpn_iou_thresh_ = tf.placeholder_with_default(0.5, [], name='rpn_iou_thresh')
    rpn_logits_ = predictions['rpn_objectness_predictions_with_background']
    rpn_logits_ = tf.reshape(rpn_logits_, (-1, rpn_logits_.shape[-1]))
    rpn_iou_ = tf.get_default_graph().get_tensor_by_name('Loss/RPNLoss/Compare/IOU/Select:0')
    rpn_iou_ = tf.reshape(rpn_iou_, (-1,))
    rpn_targets_ = tf.cast(rpn_iou_ >= rpn_iou_thresh_, tf.float32)

    # RPN Background Loss
    rpn_background_weight_ = tf.placeholder_with_default(0.0, [], name='rpn_background_weight')

    rpn_background_logits_ = rpn_logits_[:, 0]
    rpn_background_loss_ = rpn_background_logits_*rpn_targets_
    rpn_background_loss_ = -1*tf.reduce_sum(rpn_background_loss_) # Maximize!
    weighted_rpn_background_loss_ = tf.multiply(rpn_background_loss_, rpn_background_weight_, name='rpn_background_loss')

    # RPN Foreground Loss
    rpn_foreground_weight_ = tf.placeholder_with_default(0.0, [], name='rpn_foreground_weight')

    rpn_foreground_logits_ = rpn_logits_[:, 1]
    rpn_foreground_loss_ = rpn_foreground_logits_*rpn_targets_
    rpn_foreground_loss_ = tf.reduce_sum(rpn_foreground_loss_)
    weighted_rpn_foreground_loss_ = tf.multiply(rpn_foreground_loss_, rpn_foreground_weight_, name='rpn_foreground_loss')

    # RPN CW Loss (un/targeted foreground -> background)
    rpn_cw_weight_ = tf.placeholder_with_default(0.0, [], name='rpn_cw_weight')
    rpn_cw_conf_ = tf.placeholder_with_default(0.0, [], name='rpn_cw_conf')

    rpn_cw_loss_ = tf.nn.relu(rpn_foreground_logits_ - rpn_background_logits_ + rpn_cw_conf_)
    rpn_cw_loss_ = rpn_cw_loss_*rpn_targets_
    rpn_cw_loss_ = tf.reduce_sum(rpn_cw_loss_)
    weighted_rpn_cw_loss_ = tf.multiply(rpn_cw_loss_, rpn_cw_weight_, name='rpn_cw_loss')

    # Similiary Loss
    sim_weight_ = tf.placeholder_with_default(0.0, [], name='sim_weight')
    initial_textures_ = tf.get_default_graph().get_tensor_by_name('initial_textures:0')
    sim_loss_ = tf.nn.l2_loss(initial_textures_ - textures)
    weighted_sim_loss_ = tf.multiply(sim_loss_, sim_weight_, name='sim_loss')

    loss_ = tf.add_n([weighted_box_cls_loss_, weighted_box_loc_loss_,
                      weighted_rpn_cls_loss_, weighted_rpn_loc_loss_,
                      weighted_box_victim_loss_, weighted_box_target_loss_,
                      weighted_box_victim_cw_loss_, weighted_box_target_cw_loss_,
                      weighted_rpn_foreground_loss_, weighted_rpn_background_loss_,
                      weighted_rpn_cw_loss_,
                      weighted_sim_loss_], name='loss')

    # Support large batch accumulation metrics
    total_box_cls_loss_, update_box_cls_loss_ = tf.metrics.mean(losses['Loss/BoxClassifierLoss/classification_loss'])
    total_box_loc_loss_, update_box_loc_loss_ = tf.metrics.mean(losses['Loss/BoxClassifierLoss/localization_loss'])
    total_rpn_cls_loss_, update_rpn_cls_loss_ = tf.metrics.mean(losses['Loss/RPNLoss/objectness_loss'])
    total_rpn_loc_loss_, update_rpn_loc_loss_ = tf.metrics.mean(losses['Loss/RPNLoss/localization_loss'])
    total_box_target_loss_, update_box_target_loss_ = tf.metrics.mean(box_target_loss_)
    total_box_victim_loss_, update_box_victim_loss_ = tf.metrics.mean(box_victim_loss_)
    total_box_target_cw_loss_, update_box_target_cw_loss_ = tf.metrics.mean(box_target_cw_loss_)
    total_box_victim_cw_loss_, update_box_victim_cw_loss_ = tf.metrics.mean(box_victim_cw_loss_)
    total_rpn_foreground_loss_, update_rpn_foreground_loss_ = tf.metrics.mean(rpn_foreground_loss_)
    total_rpn_background_loss_, update_rpn_background_loss_ = tf.metrics.mean(rpn_background_loss_)
    total_rpn_cw_loss_, update_rpn_cw_loss_ = tf.metrics.mean(rpn_cw_loss_)
    total_sim_loss_, update_sim_loss_ = tf.metrics.mean(sim_loss_)

    total_weighted_box_cls_loss_ = tf.multiply(total_box_cls_loss_, box_cls_weight_, name='total_box_cls_loss')
    total_weighted_box_loc_loss_ = tf.multiply(total_box_loc_loss_, box_loc_weight_, name='total_box_loc_loss')
    total_weighted_rpn_cls_loss_ = tf.multiply(total_rpn_cls_loss_, rpn_cls_weight_, name='total_rpn_cls_loss')
    total_weighted_rpn_loc_loss_ = tf.multiply(total_rpn_loc_loss_, rpn_loc_weight_, name='total_rpn_loc_loss')
    total_weighted_box_target_loss_ = tf.multiply(total_box_target_loss_, box_target_weight_, name='total_box_target_loss')
    total_weighted_box_victim_loss_ = tf.multiply(total_box_victim_loss_, box_victim_weight_, name='total_box_victim_loss')
    total_weighted_box_target_cw_loss_ = tf.multiply(total_box_target_cw_loss_, box_target_cw_weight_, name='total_box_target_cw_loss')
    total_weighted_box_victim_cw_loss_ = tf.multiply(total_box_victim_cw_loss_, box_victim_cw_weight_, name='total_box_victim_cw_loss')
    total_weighted_rpn_background_loss_ = tf.multiply(total_rpn_background_loss_, rpn_background_weight_, name='total_rpn_background_loss')
    total_weighted_rpn_foreground_loss_ = tf.multiply(total_rpn_foreground_loss_, rpn_foreground_weight_, name='total_rpn_foreground_loss')
    total_weighted_rpn_cw_loss_ = tf.multiply(total_rpn_cw_loss_, rpn_cw_weight_, name='total_rpn_cw_loss')
    total_weighted_sim_loss_ = tf.multiply(total_sim_loss_, sim_weight_, name='total_sim_loss')

    total_loss_ = tf.add_n([total_weighted_box_cls_loss_, total_weighted_box_loc_loss_,
                            total_weighted_rpn_cls_loss_, total_weighted_rpn_loc_loss_,
                            total_weighted_box_target_loss_, total_weighted_box_victim_loss_,
                            total_weighted_box_target_cw_loss_, total_weighted_box_victim_cw_loss_,
                            total_weighted_rpn_foreground_loss_, total_weighted_rpn_background_loss_,
                            total_weighted_rpn_cw_loss_,
                            total_weighted_sim_loss_], name='total_loss')

    #tf.summary.scalar('losses/box_cls_weight', box_cls_weight_)
    tf.summary.scalar('losses/box_cls_loss', total_box_cls_loss_)
    tf.summary.scalar('losses/box_cls_weighted_loss', total_weighted_box_cls_loss_)

    #tf.summary.scalar('losses/box_loc_weight', box_loc_weight_)
    tf.summary.scalar('losses/box_loc_loss', total_box_loc_loss_)
    tf.summary.scalar('losses/box_loc_weighted_loss', total_weighted_box_loc_loss_)

    #tf.summary.scalar('losses/rpn_cls_weight', rpn_cls_weight_)
    tf.summary.scalar('losses/rpn_cls_loss', total_rpn_cls_loss_)
    tf.summary.scalar('losses/rpn_cls_weighted_loss', total_weighted_rpn_cls_loss_)

    #tf.summary.scalar('losses/rpn_loc_weight', rpn_loc_weight_)
    tf.summary.scalar('losses/rpn_loc_loss', total_rpn_loc_loss_)
    tf.summary.scalar('losses/rpn_loc_weighted_loss', total_weighted_rpn_loc_loss_)

    #tf.summary.scalar('losses/box_target_weight', box_target_weight_)
    tf.summary.scalar('losses/box_target_loss', total_box_target_loss_)
    tf.summary.scalar('losses/box_target_weighted_loss', total_weighted_box_target_loss_)

    #tf.summary.scalar('losses/box_victim_weight', box_victim_weight_)
    tf.summary.scalar('losses/box_victim_loss', total_box_victim_loss_)
    tf.summary.scalar('losses/box_victim_weighted_loss', total_weighted_box_victim_loss_)

    #tf.summary.scalar('losses/box_target_cw_weight', box_target_cw_weight_)
    #tf.summary.scalar('losses/box_target_cw_conf', box_target_cw_conf_)
    tf.summary.scalar('losses/box_target_cw_loss', total_box_target_cw_loss_)
    tf.summary.scalar('losses/box_target_cw_weighted_loss', total_weighted_box_target_cw_loss_)

    #tf.summary.scalar('losses/box_victim_cw_weight', box_victim_cw_weight_)
    #tf.summary.scalar('losses/box_victim_cw_conf', box_victim_cw_conf_)
    tf.summary.scalar('losses/box_victim_cw_loss', total_box_victim_cw_loss_)
    tf.summary.scalar('losses/box_victim_cw_weighted_loss', total_weighted_box_victim_cw_loss_)

    #tf.summary.scalar('losses/rpn_foreground_weight', rpn_foreground_weight_)
    tf.summary.scalar('losses/rpn_foreground_loss', total_rpn_foreground_loss_)
    tf.summary.scalar('losses/rpn_foreground_weighted_loss', total_weighted_rpn_foreground_loss_)

    #tf.summary.scalar('losses/rpn_background_weight', rpn_background_weight_)
    tf.summary.scalar('losses/rpn_background_loss', total_rpn_background_loss_)
    tf.summary.scalar('losses/rpn_background_weighted_loss', total_weighted_rpn_background_loss_)

    #tf.summary.scalar('losses/rpn_cw_weight', rpn_cw_weight_)
    #tf.summary.scalar('losses/rpn_cw_conf', rpn_cw_conf_)
    tf.summary.scalar('losses/rpn_cw_loss', total_rpn_cw_loss_)
    tf.summary.scalar('losses/rpn_cw_weighted_loss', total_weighted_rpn_cw_loss_)

    tf.summary.scalar('losses/sim_loss', total_sim_loss_)
    tf.summary.scalar('losses/sim_weighted_loss', total_weighted_sim_loss_)

    tf.summary.scalar('loss/loss', total_loss_)

    learning_rate_ = tf.placeholder(tf.float32, [], name='learning_rate')
    #tf.summary.scalar('hyperparameters/learning_rate', learning_rate_)

    momentum_ = tf.placeholder(tf.float32, [], name='momentum')
    #tf.summary.scalar('hyperparameters/momentum', momentum_)

    decay_ = tf.placeholder(tf.float32, [], name='decay')
    #tf.summary.scalar('hyperparameters/decay', decay_)

    optimizer = None
    if optimizer_name == 'gd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate_)
    elif optimizer_name == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate_, momentum=momentum_)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate_, momentum=momentum_, decay=decay_)
    elif optimizer_name == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate_)
    else:
        raise Exception("Unknown optimizer: {args.optimizer}")

    global_step_ = tf.train.get_or_create_global_step()
    grad_ = optimizer.compute_gradients(loss_, var_list=[textures_var])[0][0]
    grad_ = grad_

    # Create variable to store grad
    grad_total_ = tf.Variable(tf.zeros_like(textures_var), trainable=False, name='grad_total', collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.LOCAL_VARIABLES])
    grad_count_ = tf.Variable(0., trainable=False, name='grad_count', collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.LOCAL_VARIABLES])

    update_grad_total_ = tf.assign_add(grad_total_, grad_)
    update_grad_count_ = tf.assign_add(grad_count_, 1.)

    grad_ = tf.div(grad_total_, tf.maximum(grad_count_, 1.), name='grad')
    if clip:
        grad_ = tf.sign(grad_)

    accum_grad_op_ = tf.group([update_grad_total_, update_grad_count_], name='accum_op')

    grad_vec_ = tf.reshape(grad_, [-1], name='grad_vec')
    tf.summary.histogram('gradients/all', grad_vec_)

    grad_l1_ = tf.identity(tf.norm(grad_vec_, ord=1), name='grad_l1')
    tf.summary.scalar('gradients/all_l1', grad_l1_)

    grad_l2_ = tf.identity(tf.norm(grad_vec_, ord=2), name='grad_l2')
    tf.summary.scalar('gradients/all_l2', grad_l2_)

    grad_linf_ = tf.identity(tf.norm(grad_vec_, ord=np.inf), name='grad_linf')
    tf.summary.scalar('gradients/all_linf', grad_linf_)

    attack_op_ = optimizer.apply_gradients([(grad_, textures_var)], global_step=global_step_, name='attack_op')

    update_losses_ = tf.stack([update_box_cls_loss_, update_box_loc_loss_,
                               update_rpn_cls_loss_, update_rpn_loc_loss_,
                               update_box_target_loss_, update_box_victim_loss_,
                               update_box_target_cw_loss_, update_box_victim_cw_loss_,
                               update_rpn_foreground_loss_, update_rpn_background_loss_,
                               update_rpn_cw_loss_,
                               update_sim_loss_],
                              name='update_losses_op')

    losses_summary_ = tf.summary.merge_all()

    return victim_class_, target_class_, losses_summary_

def create_evaluation(victim_class, target_class, textures, textures_masks, input_images, detections):
    # Extract proposal, target, and victim metrics
    # Average Precision
    proposal_average_precision_var_ = tf.Variable(0.0, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    proposal_average_precision_ = tf.identity(proposal_average_precision_var_, name='proposal_average_precision')
    tf.assign(proposal_average_precision_var_, proposal_average_precision_, name='set_proposal_average_precision')

    target_average_precision_var_ = tf.Variable(0.0, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    target_average_precision_ = tf.identity(target_average_precision_var_, name='target_average_precision')
    tf.assign(target_average_precision_var_, target_average_precision_, name='set_target_average_precision')

    victim_average_precision_var_ = tf.Variable(0.0, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    victim_average_precision_ = tf.identity(victim_average_precision_var_, name='victim_average_precision')
    tf.assign(victim_average_precision_var_, victim_average_precision_, name='set_victim_average_precision')

    # Correct Localizations
    proposal_corloc_var_ = tf.Variable(0.0, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    proposal_corloc_ = tf.identity(proposal_corloc_var_, name='proposal_corloc')
    tf.assign(proposal_corloc_var_, proposal_corloc_, name='set_proposal_corloc')

    target_corloc_var_ = tf.Variable(0.0, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    target_corloc_ = tf.identity(target_corloc_var_, name='target_corloc')
    tf.assign(target_corloc_var_, target_corloc_, name='set_target_corloc')

    victim_corloc_var_ = tf.Variable(0.0, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    victim_corloc_ = tf.identity(victim_corloc_var_, name='victim_corloc')
    tf.assign(victim_corloc_var_, victim_corloc_, name='set_victim_corloc')

    # Create summaries of metrics
    metrics_summary_ = tf.summary.merge([tf.summary.scalar('metrics/proposal_average_precision', proposal_average_precision_),
                                         tf.summary.scalar('metrics/target_average_precision', target_average_precision_),
                                         tf.summary.scalar('metrics/victim_average_precision', victim_average_precision_),
                                         tf.summary.scalar('metrics/proposal_corloc', proposal_corloc_),
                                         tf.summary.scalar('metrics/target_corloc', target_corloc_),
                                         tf.summary.scalar('metrics/victim_corloc', victim_corloc_),
                                        ], name='metrics_summary')

    # TODO: Add other_average_precision

    # Create texture image summary
    textures = tf.fake_quant_with_min_max_args(textures, min=0., max=1., num_bits=8)
    rgb_textures_ = tf.image.convert_image_dtype(textures, tf.uint8, saturate=True, name='rgb_textures')

    rgba_textures_ = tf.concat([textures, textures_masks], axis=-1)
    rgba_textures_ = tf.image.convert_image_dtype(rgba_textures_, tf.uint8, saturate=True, name='rgba_textures')

    texture_summary_ = tf.summary.image('texture', rgba_textures_, max_outputs=rgba_textures_.shape[0])

    # TODO: In newer version of Tensorflow summaries can have names so these don't have to be returned
    return metrics_summary_, texture_summary_

def batch_accumulate(sess, feed_dict, count, batch_size, dict_or_func, detections, predictions, categories):
    # Accumulate and update losses and metrics for each batch
    fetches = {'accum_op': 'accum_op', 'update_losses_op': 'update_losses_op'}
    if detections is not None:
        fetches[DetectionResultFields.detection_boxes] = detections[DetectionResultFields.detection_boxes]
        fetches[DetectionResultFields.detection_scores] = detections[DetectionResultFields.detection_scores]
        fetches[DetectionResultFields.detection_classes] = detections[DetectionResultFields.detection_classes]
    if predictions is not None:
        fetches['proposal_boxes_normalized'] = predictions['proposal_boxes_normalized']
        fetches['proposal_scores'] = 'BatchMultiClassNonMaxSuppression/map/TensorArrayStack_1/TensorArrayGatherV3:0'

    # Extract stuff from feed_dict
    victim_score_thresh = 0.01 # 1/num_classes is a good value
    target_score_thresh = 0.01 # 1/num_classes is a good value

    target_class = feed_dict['target_class:0']
    victim_class = feed_dict['victim_class:0']

    iou_thresh = 0.5

    proposal_label = 'foreground'
    target_label = next(filter(lambda category: category['id'] == target_class, categories))['name']
    victim_label = next(filter(lambda category: category['id'] == victim_class, categories))['name']

    # Create evaluators and set average precisions
    # TODO: parameterize matching_iou_threshold
    proposal_evaluator = ObjectDetectionEvaluator([{'id': 1, 'name': proposal_label}], matching_iou_threshold=iou_thresh, evaluate_corlocs=True)
    victim_evaluator = ObjectDetectionEvaluator(categories, matching_iou_threshold=iou_thresh, evaluate_corlocs=True)
    target_evaluator = ObjectDetectionEvaluator(categories, matching_iou_threshold=iou_thresh, evaluate_corlocs=True)

    for i in range(0, count, batch_size):
        batch_feed_dict = feed_dict.copy()

        if i + batch_size > count:
            print('Batch size not multiple of inputs!')
            continue

        if isinstance(dict_or_func, dict):
            data = { key: dict_or_func[key][i:i+batch_size] for key in dict_or_func }
        else:
            data = dict_or_func(batch_size)

        for key, values in data.items():
            if '%d' in key: # Treat keys with '%d' in them as special
                for j in range(batch_size):
                    batch_feed_dict[key % j] = values[j]
            else:
                batch_feed_dict[key] = values

        outputs = sess.run(fetches, batch_feed_dict)

        # Add groundtruth and detections to evaluators
        for j, (proposal_boxes, proposal_scores,
                detection_boxes, detection_scores, detection_classes) in enumerate(zip(outputs['proposal_boxes_normalized'],
                                                                                       outputs['proposal_scores'],
                                                                                       outputs[DetectionResultFields.detection_boxes],
                                                                                       outputs[DetectionResultFields.detection_scores],
                                                                                       outputs[DetectionResultFields.detection_classes])):
            # Get groundtruth for current detections
            groundtruth_boxes = batch_feed_dict['groundtruth_boxes_%d:0' % j]
            box_scaler = np.array(batch_feed_dict['backgrounds:0'].shape[1:3]*2)

            groundtruth = {
                InputDataFields.groundtruth_boxes: groundtruth_boxes * box_scaler,
                InputDataFields.groundtruth_classes: np.zeros(groundtruth_boxes.shape[:1])
            }

            # Update rpn detections
            proposal_detections = {
                DetectionResultFields.detection_boxes: proposal_boxes * box_scaler,
                DetectionResultFields.detection_scores: proposal_scores,
                DetectionResultFields.detection_classes: np.ones_like(proposal_scores)
            }

            groundtruth[InputDataFields.groundtruth_classes][:] = 1
            proposal_evaluator.add_single_ground_truth_image_info(f"image{i+j}", groundtruth)
            proposal_evaluator.add_single_detected_image_info(f"image{i+j}", proposal_detections)

            # Filter victim detections
            selected_victims = detection_scores > victim_score_thresh
            victim_detections = {
                DetectionResultFields.detection_boxes: detection_boxes[selected_victims] * box_scaler,
                DetectionResultFields.detection_scores: detection_scores[selected_victims],
                DetectionResultFields.detection_classes: detection_classes[selected_victims] + 1
            }

            groundtruth[InputDataFields.groundtruth_classes][:] = victim_class
            victim_evaluator.add_single_ground_truth_image_info(f"image{i+j}", groundtruth)
            victim_evaluator.add_single_detected_image_info(f"image{i+j}", victim_detections)

            # Filter target detections
            selected_targets = detection_scores > target_score_thresh
            target_detections = {
                DetectionResultFields.detection_boxes: detection_boxes[selected_targets] * box_scaler,
                DetectionResultFields.detection_scores: detection_scores[selected_targets],
                DetectionResultFields.detection_classes: detection_classes[selected_targets] + 1
            }

            groundtruth[InputDataFields.groundtruth_classes][:] = target_class
            target_evaluator.add_single_ground_truth_image_info(f"image{i+j}", groundtruth)
            target_evaluator.add_single_detected_image_info(f"image{i+j}", target_detections)

    proposal_metrics = proposal_evaluator.evaluate()
    victim_metrics = victim_evaluator.evaluate()
    target_metrics = target_evaluator.evaluate()

    sess.run('set_proposal_average_precision', {'proposal_average_precision:0': proposal_metrics['PerformanceByCategory/AP@{}IOU/{}'.format(iou_thresh, proposal_label)]})
    sess.run('set_victim_average_precision', {'victim_average_precision:0': victim_metrics['PerformanceByCategory/AP@{}IOU/{}'.format(iou_thresh, victim_label)]})
    sess.run('set_target_average_precision', {'target_average_precision:0': target_metrics['PerformanceByCategory/AP@{}IOU/{}'.format(iou_thresh, target_label)]})

    sess.run('set_proposal_corloc', {'proposal_corloc:0': proposal_metrics['PerformanceByCategory/CorLoc@{}IOU/{}'.format(iou_thresh, proposal_label)]})
    sess.run('set_victim_corloc', {'victim_corloc:0': victim_metrics['PerformanceByCategory/CorLoc@{}IOU/{}'.format(iou_thresh, victim_label)]})
    sess.run('set_target_corloc', {'target_corloc:0': target_metrics['PerformanceByCategory/CorLoc@{}IOU/{}'.format(iou_thresh, target_label)]})

def batch_run(sess, fetches, feed_dict, batch_size, data):
    outputs = {}

    count = np.max([values.shape[0] for values in data.values()])

    for i in range(0, count, batch_size):
        batch_feed_dict = feed_dict.copy()

        if i + batch_size > count:
            print('Batch size not multiple of inputs!')
            continue

        for key, values in data.items():
            if '%d' in key: # Treat keys with '%d' in them as special
                for j in range(batch_size):
                    batch_feed_dict[key % j] = values[i+j]
            else:
                batch_feed_dict[key] = values[i:i+batch_size]

        batch_outputs = sess.run(fetches, batch_feed_dict)

        for key, value in batch_outputs.items():
            if value is None:
                continue
            if key not in outputs:
                outputs[key] = []
            if not isinstance(value, np.ndarray):
                value = np.array([value])
            outputs[key].append(value)

    for key in outputs:
        outputs[key] = np.concatenate(outputs[key])

    return outputs

def plot(data, sess, batch_size, bboxes=None, detections=None, col_wrap=5, include_textures=False, width=10, height=None, min_score_thresh=0.50, category_index=None, feed_dict={}, line_thickness=3):
    feed_dict = feed_dict.copy()

    fetches = {'input_images': 'input_images:0'}
    if detections is not None:
        fetches.update(detections)

    # For some reason, we need to supply these.
    count = np.max([batch_size] + [values.shape[0] for values in data.values()])
    if 'groundtruth_weights_%d:0' not in data:
        data['groundtruth_weights_%d:0'] = np.zeros((count, 0))
    if 'groundtruth_classes_%d:0' not in data:
        data['groundtruth_classes_%d:0'] = np.zeros((count, 0))
    if 'groundtruth_boxes_%d:0' not in data:
        data['groundtruth_boxes_%d:0'] = np.zeros((count, 0, 4))

    outputs = batch_run(sess, fetches, feed_dict, batch_size, data)

    if bboxes is not None:
        outputs['detection_boxes'] = bboxes

    ncount = outputs['input_images'].shape[0]
    if include_textures:
        textures = sess.run('rgba_textures:0')
        ncount = ncount + textures.shape[0]
    if col_wrap is None:
        col_wrap = ncount
    ncols = col_wrap
    nrows = int((ncount - 1) / ncols)+1

    aspect_ratio = float(outputs['input_images'].shape[1]) / float(outputs['input_images'].shape[2])
    if height is None:
        height = aspect_ratio * width

    fig = plt.figure(figsize=(width*ncols, height*nrows))
    fig.patch.set_facecolor('white')

    if include_textures:
        for i in range(textures.shape[0]):
            plt.subplot(nrows, ncols, 1+i)
            plt.title('Texture')
            plt.imshow(textures[i])
            plt.axis('off')
            plt.tick_params(axis='both', which='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)

    images = []

    for i, render in enumerate(outputs['input_images']):
        image = (255*render).astype(np.uint8)

        bboxes = None
        if 'detection_boxes' in outputs:
            bboxes = outputs['detection_boxes'][i]

        labels = None
        agnostic_mode = True
        if 'detection_classes' in outputs:
            labels = outputs['detection_classes'][i].astype(np.int32) + 1
            agnostic_mode = False

        scores = None
        if 'detection_scores' in outputs:
            scores = outputs['detection_scores'][i]

        if bboxes is not None:
            image = visualization_utils.visualize_boxes_and_labels_on_image_array(image, bboxes, labels, scores,
                                                                                  category_index,
                                                                                  use_normalized_coordinates=True,
                                                                                  max_boxes_to_draw=bboxes.shape[0],
                                                                                  min_score_thresh=min_score_thresh,
                                                                                  agnostic_mode=agnostic_mode,
                                                                                  skip_labels=(labels is None),
                                                                                  skip_scores=(scores is None),
                                                                                  line_thickness=line_thickness)
        if include_textures:
            i = i + textures.shape[0]
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(image)
        plt.axis('off')
        plt.tick_params(axis='both', which='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
        images.append(image)
    plt.show()

    return images
