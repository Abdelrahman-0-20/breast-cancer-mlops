"""Shared configuration for the breast-cancer MLOps pipeline.

Every threshold and split parameter lives here (never inline in the
modules) so that a single file documents the whole contract.
"""

# Determinism
SEED = 42

# Data contract
EXPECTED_N_FEATURES = 30

# Split fractions (stratified, must sum to 1.0)
TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15
TEST_FRACTION = 0.15

# Training
MAX_ITER = 1000

# Evaluation gate (mlops/evaluate.py, milestone M2)
MIN_TEST_ROC_AUC = 0.95
MIN_TEST_ACCURACY = 0.90

# Drift monitor (mlops/drift.py, milestone M2):
# drifted iff mean PSI > PSI_MEAN_THRESHOLD OR max PSI > PSI_MAX_THRESHOLD
PSI_MEAN_THRESHOLD = 0.15
PSI_MAX_THRESHOLD = 0.30
