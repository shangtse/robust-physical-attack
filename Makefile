MODEL_NAME=faster_rcnn_inception_v2_coco_2017_11_08
MODELS_DIR=models
DATA_DIR=data
LOG_DIR=logdir

OBJDET_DIR=$(shell pipenv run pip show object-detection | grep "Location:" | cut -d" " -f2)
LUCID_DIR=$(shell pipenv run pip show lucid | grep "Location:" | cut -d" " -f2)/lucid

.PHONY: help deps tensorboard 2d_targeted_attack 2d_untargeted_attack 2d_proposal_attack 2d_hybrid_targeted_attack 2d_hybrid_untargeted_attack

# Taken from https://tech.davis-hansson.com/p/make/
ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

# Taken from https://suva.sh/posts/well-documented-makefiles/
help:  ## Display this help
> @awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[1-9a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Dependencies
deps: ## Install dependencies, compile protobufs, and patch projects.
> pipenv install --three
> curl -Lo $(OBJDET_DIR)/protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
> unzip $(OBJDET_DIR)/protobuf.zip -d $(OBJDET_DIR)
> $(OBJDET_DIR)/bin/protoc -I$(OBJDET_DIR) $(OBJDET_DIR)/object_detection/protos/*.proto --python_out=$(OBJDET_DIR)
> patch -p1 -d $(OBJDET_DIR) < object_detection_api.diff
> patch -p1 -d $(LUCID_DIR) < lucid.diff

$(MODELS_DIR)/$(MODEL_NAME):
> mkdir -p $(MODELS_DIR)
> curl http://download.tensorflow.org/models/object_detection/$(MODEL_NAME).tar.gz | tar xzvf - -C $(MODELS_DIR)
> cp $(OBJDET_DIR)/object_detection/data/mscoco_label_map.pbtxt $@/label_map.pbtxt

##@ Helpers
tensorboard: ## Launch tensorboard to monitor progress.
> pipenv run tensorboard --host 127.0.0.1 --port 6006 --logdir $(LOG_DIR)

##@ Attacks
2d_targeted_attack: $(MODELS_DIR)/$(MODEL_NAME) ## Create 2d stop sign that is detected as a person.
> pipenv run python shapeshifter2d.py --verbose \
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

2d_untargeted_attack: $(MODELS_DIR)/$(MODEL_NAME) ## Create 2d stop sign that is not detected as a stop sign.
> pipenv run python shapeshifter2d.py --verbose \
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
                                      --optimizer "gd" --learning-rate 0.001 --spectral False --sign-gradients True --gray-start \
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

2d_proposal_attack: $(MODELS_DIR)/$(MODEL_NAME) ## Create 2d stop sign that is not detected.
> pipenv run python shapeshifter2d.py --verbose \
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

2d_hybrid_targeted_attack: $(MODELS_DIR)/$(MODEL_NAME) ## Create 2d stop sign that is either not detected at all or detected as a person.
> pipenv run python shapeshifter2d.py --verbose \
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
                                      --box-cls-weight 4 --box-loc-weight 0 \
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

2d_hybrid_untargeted_attack: $(MODELS_DIR)/$(MODEL_NAME) ## Create 2d stop sign that is either not detected at all or not detected as a stop sign.
> pipenv run python shapeshifter2d.py --verbose \
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
                                      --optimizer "gd" --learning-rate 0.001 --spectral False --sign-gradients True --gray-start \
                                      --rpn-loc-weight 0 --rpn-cls-weight -4 \
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

