from egrecho.utils.types import Split

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
