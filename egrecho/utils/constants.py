# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-03)

from egrecho.utils.types import Split

# If the specified file exists, the CLI will import it, providing flexibility for constructing local projects.
LOCAL_EGRECHO = "/dynamic_egrecho.py"

DATASET_META_FILENAME = "dataset_meta.json"
DEFAULT_FILES = {
    str(Split.TRAIN): "egs*train*",
    str(Split.VALIDATION): "egs*val*",
}

DEFAULT_TRAIN_FILENAME = "train.yaml"

DEFAULT_MODEL_FILENAME = "hparams.yaml"

CHECKPOINT_DIR_NAME = "checkpoints"
BEST_K_MAP_FNAME = "best_k_models.yaml"  # same level as ckpt files
CHECKPOINT_CONFIG_DIRNAME = "config"  # same level as ckpt files
DEFAULT_EXTRACTOR_FILENAME = "feature_config.yaml"  # under config
TYPE_FILENAME = "types.yaml"

MODEL_TYPE_KEY = "model_type"
EXTRACTOR_KEY = "feature_extractor_type"
