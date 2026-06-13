import numpy as np

# ---------------------------------------------------------------------------
# Biological Constants
# ---------------------------------------------------------------------------

# Sparse Distributed Representation (SDR) dimensions
N = 16384  # Total number of neurons in a layer/column
W = 64     # Number of active neurons (0.39% sparsity)

# Neuromodulation
BASE_ACH = 0.1       # Baseline Acetylcholine (baseline learning rate)
MAX_ACH = 1.0        # Max Acetylcholine (max plasticity during high surprise)
ACH_DECAY = 0.95     # How fast ACh decays back to baseline

BASE_DA = 0.0        # Baseline Dopamine
MAX_DA = 1.0         # Max Dopamine (during reward)
MIN_DA = -1.0        # Min Dopamine (during pain/punishment)
DA_DECAY = 0.90      # How fast DA decays back to baseline

# Basal Ganglia Action Selection
BABBLE_THRESHOLD = 0.2  # If max expected reward is below this, the agent babbles.
BABBLE_RATE = 0.1       # The probability of a totally random motor action even if confident (exploration).

# Synaptic Constants
MAX_PERMANENCE = 1.0
MIN_PERMANENCE = 0.0
SYNAPSE_THRESHOLD = 0.5  # Permanence must be >= this to be considered "connected"

# ---------------------------------------------------------------------------
# Data Types for Numba Compilation
# ---------------------------------------------------------------------------
SDR_TYPE = np.uint16  # Active cell indices are 0 to 16383, fits in uint16
WEIGHT_TYPE = np.float32
