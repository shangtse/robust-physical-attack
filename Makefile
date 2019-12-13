MODEL_NAME=faster_rcnn_inception_v2_coco_2017_11_08
MODELS_DIR=models
DATA_DIR=data
LOG_DIR=logdir

MODEL_URL=http://download.tensorflow.org/models/object_detection/$(MODEL_NAME).tar.gz
PROTOC_URL=https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
OBJDET_DIR=`pipenv run pip show object-detection | grep "Location:" | cut -d" " -f2`
LUCID_DIR=`pipenv run pip show lucid | grep "Location:" | cut -d" " -f2`/lucid

.PHONY: deps tensorboard 2d_targeted_attack 2d_untargeted_attack 2d_rpn_attack

deps:
	pipenv install --three
	curl -Lo $(OBJDET_DIR)/protobuf.zip $(PROTOC_URL)
	unzip $(OBJDET_DIR)/protobuf.zip -d $(OBJDET_DIR)
	$(OBJDET_DIR)/bin/protoc -I$(OBJDET_DIR) $(OBJDET_DIR)/object_detection/protos/*.proto --python_out=$(OBJDET_DIR)
	patch -p1 -d $(OBJDET_DIR) < object_detection_api.diff
	patch -p1 -d $(LUCID_DIR) < lucid.diff

$(MODELS_DIR)/$(MODEL_NAME):
	mkdir -p $(MODELS_DIR)
	curl http://download.tensorflow.org/models/object_detection/$(MODEL_NAME).tar.gz | tar xzvf - -C $(MODELS_DIR)
	cp $(OBJDET_DIR)/object_detection/data/mscoco_label_map.pbtxt $@/label_map.pbtxt

tensorboard:
	pipenv run tensorboard --host 127.0.0.1 --port 6006 --logdir $(LOG_DIR)

2d_targeted_attack: $(MODELS_DIR)/$(MODEL_NAME)
	pipenv run python shapeshifter2d.py --verbose \
                                    --model $(MODELS_DIR)/$(MODEL_NAME) \
                                    --backgrounds $(DATA_DIR)/background.png $(DATA_DIR)/background2.png \
                                    --objects $(DATA_DIR)/stop_sign_object.png \
                                    --textures $(DATA_DIR)/stop_sign_mask.png \
                                    --textures-masks $(DATA_DIR)/stop_sign_mask.png \
                                    --victim "stop sign" --target "person" \
                                    --logdir $(LOG_DIR)/$@ \
                                    --object-roll-min -15 --object-roll-max 15 \
                                    --object-x-min -500 --object-x-max 500 \
                                    --object-y-min -200 --object-y-max 200 \
                                    --object-z-min 0.1 --object-z-max 1.0 \
                                    --texture-roll-min 0 --texture-roll-max 0 \
                                    --texture-x-min 0 --texture-x-max 0 \
                                    --texture-y-min 0 --texture-y-max 0 \
                                    --texture-z-min 1.0 --texture-z-max 1.0 \
                                    --optimizer "gd" --learning-rate 0.001 --spectral False --sign-gradients True \
                                    --rpn-loc-weight 0 --rpn-cls-weight 0 \
                                    --rpn-cw-weight 0 --rpn-cw-conf 0 \
                                    --box-cls-weight 1 --box-loc-weight 0 \
                                    --box-victim-cw-weight 0 --box-victim-cw-conf 0 \
                                    --box-target-cw-weight 0 --box-target-cw-conf 0 \
                                    --box-victim-weight 0 --box-target-weight 0 \
                                    --sim-weight 0.0001 \
                                    --image-multiplicative-channel-noise-min 0.7 --image-multiplicative-channel-noise-max 1.3 \
                                    --image-additive-channel-noise-min -0.15 --image-additive-channel-noise-max 0.15 \
                                    --image-multiplicative-pixel-noise-min 0.5 --image-multiplicative-pixel-noise-max 2.0 \
                                    --image-additive-pixel-noise-min -0.15 --image-additive-pixel-noise-max 0.15 \
                                    --image-gaussian-noise-stddev-min 0.0 --image-gaussian-noise-stddev-max 0.1 \
                                    --batch-size 1 --train-batch-size 10 --test-batch-size 10

2d_untargeted_attack: $(MODELS_DIR)/$(MODEL_NAME)
	pipenv run python shapeshifter2d.py --verbose \
                                    --model $(MODELS_DIR)/$(MODEL_NAME) \
                                    --backgrounds $(DATA_DIR)/background.png $(DATA_DIR)/background2.png \
                                    --objects $(DATA_DIR)/stop_sign_object.png \
                                    --textures $(DATA_DIR)/stop_sign_mask.png \
                                    --textures-masks $(DATA_DIR)/stop_sign_mask.png \
                                    --victim "stop sign" --target "stop sign" \
                                    --logdir $(LOG_DIR)/$@ \
                                    --object-roll-min -15 --object-roll-max 15 \
                                    --object-x-min -500 --object-x-max 500 \
                                    --object-y-min -200 --object-y-max 200 \
                                    --object-z-min 0.1 --object-z-max 1.0 \
                                    --texture-roll-min 0 --texture-roll-max 0 \
                                    --texture-x-min 0 --texture-x-max 0 \
                                    --texture-y-min 0 --texture-y-max 0 \
                                    --texture-z-min 1.0 --texture-z-max 1.0 \
                                    --optimizer "gd" --learning-rate 0.001 --spectral False --sign-gradients True \
                                    --rpn-loc-weight 0 --rpn-cls-weight 0 \
                                    --rpn-cw-weight 0 --rpn-cw-conf 0 \
                                    --box-cls-weight -1 --box-loc-weight 0 \
                                    --box-victim-cw-weight 0 --box-victim-cw-conf 0 \
                                    --box-target-cw-weight 0 --box-target-cw-conf 0 \
                                    --box-victim-weight 0 --box-target-weight 0 \
                                    --sim-weight 0.0001 \
                                    --image-multiplicative-channel-noise-min 0.7 --image-multiplicative-channel-noise-max 1.3 \
                                    --image-additive-channel-noise-min -0.15 --image-additive-channel-noise-max 0.15 \
                                    --image-multiplicative-pixel-noise-min 0.5 --image-multiplicative-pixel-noise-max 2.0 \
                                    --image-additive-pixel-noise-min -0.15 --image-additive-pixel-noise-max 0.15 \
                                    --image-gaussian-noise-stddev-min 0.0 --image-gaussian-noise-stddev-max 0.1 \
                                    --batch-size 1 --train-batch-size 10 --test-batch-size 10

2d_rpn_attack: $(MODELS_DIR)/$(MODEL_NAME)
	pipenv run python shapeshifter2d.py --verbose \
                                    --model $(MODELS_DIR)/$(MODEL_NAME) \
                                    --backgrounds $(DATA_DIR)/background.png $(DATA_DIR)/background2.png \
                                    --objects $(DATA_DIR)/stop_sign_object.png \
                                    --textures $(DATA_DIR)/stop_sign_mask.png \
                                    --textures-masks $(DATA_DIR)/stop_sign_mask.png \
                                    --victim "stop sign" --target "person" \
                                    --logdir $(LOG_DIR)/$@ \
                                    --object-roll-min -15 --object-roll-max 15 \
                                    --object-x-min -500 --object-x-max 500 \
                                    --object-y-min -200 --object-y-max 200 \
                                    --object-z-min 0.1 --object-z-max 1.0 \
                                    --texture-roll-min 0 --texture-roll-max 0 \
                                    --texture-x-min 0 --texture-x-max 0 \
                                    --texture-y-min 0 --texture-y-max 0 \
                                    --texture-z-min 1.0 --texture-z-max 1.0 \
                                    --optimizer "gd" --learning-rate 0.001 --spectral False --sign-gradients True --gray-start \
                                    --rpn-loc-weight 0 --rpn-cls-weight -1 \
                                    --rpn-cw-weight 0 --rpn-cw-conf 0 \
                                    --box-cls-weight 0 --box-loc-weight 0 \
                                    --box-victim-cw-weight 0 --box-victim-cw-conf 0 \
                                    --box-target-cw-weight 0 --box-target-cw-conf 0 \
                                    --box-victim-weight 0 --box-target-weight 0 \
                                    --sim-weight 0.00005 \
                                    --image-multiplicative-channel-noise-min 0.7 --image-multiplicative-channel-noise-max 1.3 \
                                    --image-additive-channel-noise-min -0.15 --image-additive-channel-noise-max 0.15 \
                                    --image-multiplicative-pixel-noise-min 0.5 --image-multiplicative-pixel-noise-max 2.0 \
                                    --image-additive-pixel-noise-min -0.15 --image-additive-pixel-noise-max 0.15 \
                                    --image-gaussian-noise-stddev-min 0.0 --image-gaussian-noise-stddev-max 0.1 \
                                    --batch-size 1 --train-batch-size 10 --test-batch-size 10

