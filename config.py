import os

# Model & data
MODEL_NAME = "google/mt5-small"
PREFIX = "fix grammar: "
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
NUM_TRAIN_EPOCHS = 5
WEIGHT_DECAY = 0.01
SEED = 42

# Paths (relative to notebook/workspace)
CHECKPOINT_DIR = os.path.join("models", "checkpoints")
FINAL_MODEL_DIR = os.path.join("models", "final_model")
OUTPUTS_DIR = "outputs"
DATA_PATH = "dataset_final_processed_v2.jsonl"

# Misc defaults
FP16 = False
MAX_PREDICTION_LENGTH = 128
