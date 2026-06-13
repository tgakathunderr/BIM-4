import numpy as np
from bim4.config import BASE_ACH, MAX_ACH, ACH_DECAY, BASE_DA, MAX_DA, MIN_DA, DA_DECAY

class Brainstem:
    """
    The chemical bath of the digital species.
    Tracks global neuromodulators that dictate learning rates and action gating.
    """
    def __init__(self):
        self.ach = BASE_ACH  # Acetylcholine: Surprise/Plasticity
        self.da = BASE_DA    # Dopamine: Reward/Action selection
        
        # Habituation tracking for ACh
        self.recent_surprises = []
        self.habituation_window = 10
        
    def process_tick(self, surprise_signal: float, reward_signal: float):
        """
        Updates chemical levels based on the current tick's events.
        surprise_signal: 0.0 to 1.0 (from Cortex Jaccard distance)
        reward_signal: -1.0 (Pain) to 1.0 (Reward) (from REPL tags)
        """
        # --- Dopamine (Reward/Pain) ---
        if reward_signal != 0.0:
            # External reward/pain spikes dopamine immediately
            self.da = np.clip(self.da + reward_signal, MIN_DA, MAX_DA)
        else:
            # Decay towards baseline
            self.da = self.da * DA_DECAY + BASE_DA * (1 - DA_DECAY)
            
        # --- Acetylcholine (Surprise/Plasticity) ---
        # Habituation: if we've been surprised by the EXACT same amount repeatedly, it's not actually surprising
        self.recent_surprises.append(surprise_signal)
        if len(self.recent_surprises) > self.habituation_window:
            self.recent_surprises.pop(0)
            
        # Calculate variance in recent surprises to prevent getting stuck in high-ACh states on pure noise
        if len(self.recent_surprises) == self.habituation_window:
            variance = np.var(self.recent_surprises)
            if variance < 0.01 and surprise_signal > 0.5:
                # High surprise but low variance means it's predictable noise. Force habituation.
                surprise_signal *= 0.5
                
        # Update ACh
        if surprise_signal > BASE_ACH:
            self.ach = np.clip(self.ach + surprise_signal * 0.5, BASE_ACH, MAX_ACH)
        else:
            # Decay towards baseline
            self.ach = self.ach * ACH_DECAY + BASE_ACH * (1 - ACH_DECAY)
            
        return self.ach, self.da

    def get_state(self):
        return {"ACh": round(self.ach, 3), "DA": round(self.da, 3)}
