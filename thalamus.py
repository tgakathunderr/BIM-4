import numpy as np
from bim4.config import N, W, SDR_TYPE

class Thalamus:
    """
    The sensory gateway.
    Converts raw environmental bytes into Sparse Distributed Representations (SDRs).
    Also parses explicit tags like [REWARD] out of the sensory stream.
    """
    def __init__(self):
        # Create a fixed, deterministic hash map for all 256 possible bytes.
        # This ensures the same byte always produces the exact same SDR.
        self.byte_to_sdr = {}
        
        # Seed it so it's consistent across runs for the same 'species'
        rng = np.random.RandomState(42) 
        
        for b in range(256):
            # Choose W unique indices out of N
            indices = rng.choice(N, size=W, replace=False)
            indices.sort()
            self.byte_to_sdr[b] = np.array(indices, dtype=SDR_TYPE)

    def encode(self, byte_val: int) -> np.ndarray:
        """
        Takes a raw byte (0-255) and returns its SDR (array of active indices).
        """
        assert 0 <= byte_val <= 255, f"Byte value {byte_val} must be 0-255"
        return self.byte_to_sdr[byte_val]
        
    def decode(self, sdr: np.ndarray) -> int:
        """
        Finds the byte that most closely matches the given SDR (used for motor output decoding).
        Returns the byte (0-255) with the highest overlap.
        """
        best_byte = 0
        max_overlap = -1
        
        for b, target_sdr in self.byte_to_sdr.items():
            overlap = np.intersect1d(sdr, target_sdr, assume_unique=True).size
            if overlap > max_overlap:
                max_overlap = overlap
                best_byte = b
                
        return best_byte

    def parse_sensory_stream(self, input_string: str):
        """
        Separates explicit tags from raw sensory text.
        Returns (clean_text, reward_signal, sleep_flag)
        """
        reward_signal = 0.0
        sleep_flag = False
        
        # Extract tags
        if "[REWARD]" in input_string:
            reward_signal = 1.0
            input_string = input_string.replace("[REWARD]", "")
            
        if "[PAIN]" in input_string:
            reward_signal = -1.0
            input_string = input_string.replace("[PAIN]", "")
            
        if "[SLEEP]" in input_string:
            sleep_flag = True
            input_string = input_string.replace("[SLEEP]", "")
            
        return input_string.strip(), reward_signal, sleep_flag
