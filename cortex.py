import numpy as np
from numba import njit
from bim4.config import N, W, SDR_TYPE

@njit
def compute_prediction(weights, active_indices):
    """
    Given active indices, sums the permanences to all other cells.
    Returns the indices of the top W cells.
    """
    scores = np.zeros(N, dtype=np.int32)
    for idx in active_indices:
        scores += weights[idx, :]
        
    # Get top W indices. 
    # argsort is supported by numba. Negate scores for descending order.
    top_w_indices = np.argsort(-scores)[:W]
    
    # If all scores are 0, argsort just returns 0..W-1. 
    # We should return empty if there is absolutely no prediction strength.
    if scores[top_w_indices[0]] == 0:
        return np.zeros(0, dtype=SDR_TYPE)
        
    return top_w_indices.astype(SDR_TYPE)

@njit
def update_synapses(weights, prev_sdr, current_sdr, ach_rate):
    """
    Hebbian learning.
    ach_rate: 0 to 100 (integer scaled)
    """
    # Create boolean mask for fast checking
    is_current = np.zeros(N, dtype=np.bool_)
    for idx in current_sdr:
        is_current[idx] = True
        
    for p_idx in prev_sdr:
        for i in range(N):
            if is_current[i]:
                # Strengthen (LTP)
                new_val = weights[p_idx, i] + ach_rate
                if new_val > 100:
                    weights[p_idx, i] = 100
                else:
                    weights[p_idx, i] = new_val
            else:
                # Weak decay (LTD) to unlearn false predictions
                if weights[p_idx, i] > 0:
                    weights[p_idx, i] -= 1

class Cortex:
    """
    Numba-accelerated sparse predictive sequence memory.
    Maintains a 1-step temporal sequence memory of SDRs.
    """
    def __init__(self):
        # 268 MB dense uint8 matrix representing permanences (0-100)
        # 0 = no synapse, 100 = max strength
        self.weights = np.zeros((N, N), dtype=np.uint8)
        self.prev_sdr = np.array([], dtype=SDR_TYPE)
        self.predicted_sdr = np.array([], dtype=SDR_TYPE)

    def process_sdr(self, current_sdr: np.ndarray, ach: float) -> tuple[np.ndarray, float]:
        """
        Processes a new sensory SDR.
        Returns:
            - The NEW prediction for the *next* timestep.
            - The surprise of the *current* SDR compared to the old prediction.
        """
        # 1. Calculate surprise of the current input
        surprise = 1.0
        if len(self.predicted_sdr) > 0 and len(current_sdr) > 0:
            intersection = np.intersect1d(self.predicted_sdr, current_sdr, assume_unique=True).size
            union = len(self.predicted_sdr) + len(current_sdr) - intersection
            if union > 0:
                surprise = 1.0 - (intersection / union)
                
        # 2. Learn (update synapses from prev_sdr to current_sdr)
        if len(self.prev_sdr) > 0 and len(current_sdr) > 0 and ach > 0.01:
            ach_int = int(ach * 10) # scale to 1-10 max increase per step
            update_synapses(self.weights, self.prev_sdr, current_sdr, ach_int)
            
        # 3. Predict next
        if len(current_sdr) > 0:
            self.predicted_sdr = compute_prediction(self.weights, current_sdr)
        else:
            self.predicted_sdr = np.array([], dtype=SDR_TYPE)
            
        self.prev_sdr = current_sdr.copy()
        
        return self.predicted_sdr, surprise

    def force_predict(self, sdr: np.ndarray) -> np.ndarray:
        """ Used by Basal Ganglia / Sleep to predict without advancing state """
        if len(sdr) == 0:
            return np.array([], dtype=SDR_TYPE)
        return compute_prediction(self.weights, sdr)
