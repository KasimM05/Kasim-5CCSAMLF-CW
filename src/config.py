from pathlib import Path

RANDOM_STATE = 42
TARGET_COL = "outcome"

TRAIN_FILE = "CW1_train.csv"
TEST_FILE = "CW1_test.csv"

SUBMISSION_DIR = Path("submissions")
SUBMISSION_NAME = "CW1_submission_K24060083.csv"

# Best params from tuning (stage 2)
HGB_FIXED_PARAMS = {
    "learning_rate": 0.039,
    "max_iter": 300,
    "max_depth": 3,
    "max_leaf_nodes": 255,
    "min_samples_leaf": 10,
    "l2_regularization": 1.0,
}
