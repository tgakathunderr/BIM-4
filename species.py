import numpy as np
from bim4.brainstem import Brainstem
from bim4.thalamus import Thalamus
from bim4.basal_ganglia import BasalGanglia
from bim4.cortex import Cortex
from bim4.hippocampus import Hippocampus

class DigitalSpecies:
    """
    The unified biological substrate.
    Wires all organs together and exposes a simple interaction interface.
    """
    def __init__(self):
        self.brainstem = Brainstem()
        self.thalamus = Thalamus()
        self.cortex = Cortex()
        self.basal_ganglia = BasalGanglia()
        self.hippocampus = Hippocampus()

    def tick(self, sensory_byte: int, reward_signal: float = 0.0) -> tuple[int, bool]:
        """
        Runs one global brain tick.
        Returns the (chosen_motor_byte, is_babbling).
        """
        # 1. Thalamus encodes
        current_sdr = self.thalamus.encode(sensory_byte)
        
        # 2. Cortex predicts next, and calculates surprise
        predicted_sdr, surprise = self.cortex.process_sdr(current_sdr, self.brainstem.ach)
        
        # 3. Brainstem updates chemicals
        ach, da = self.brainstem.process_tick(surprise, reward_signal)
        
        # 4. Basal Ganglia evaluates current situation and proposes action
        expected_rewards = self.basal_ganglia.score_actions(current_sdr)
        chosen_action, is_babbling = self.basal_ganglia.select_action(expected_rewards)
        
        # 5. Basal Ganglia learns from reward
        self.basal_ganglia.learn(current_sdr, chosen_action, da)
        
        # 6. Hippocampus stores the episode
        self.hippocampus.store(current_sdr, chosen_action, da, ach)
        
        return chosen_action, is_babbling

    def sleep(self):
        """
        Offline consolidation replay.
        """
        episodes = self.hippocampus.get_dream_sequence()
        # High ACh during sleep to lock in memories rapidly
        ach_override = 1.0 
        
        # Temporarily reset cortical state for sleep sequence
        self.cortex.prev_sdr = np.array([], dtype=np.uint16)
        
        for (sdr, action, da, _) in episodes:
            # Replay cortical sequence
            self.cortex.process_sdr(sdr, ach_override)
            # Replay BG reward
            self.basal_ganglia.learn(sdr, action, da)
            
        self.hippocampus.clear()
        
    def interact(self, text_input: str) -> str:
        """ Processes a full string of text/tags and yields output characters. """
        clean_text, reward_signal, sleep_flag = self.thalamus.parse_sensory_stream(text_input)
        
        if sleep_flag:
            self.sleep()
            return "[Agent fell asleep and consolidated memories]"
            
        outputs = []
        
        # If user gave no text, but just a reward, process reward on a space character
        bytes_to_process = [ord(c) for c in clean_text]
        if len(bytes_to_process) == 0 and reward_signal != 0:
            bytes_to_process = [32] 
            
        for i, b in enumerate(bytes_to_process):
            # Only apply the reward signal to the LAST byte of the sequence
            r = reward_signal if i == len(bytes_to_process) - 1 else 0.0
            
            action, is_babbling = self.tick(b, r)
            outputs.append(chr(action))
            
        return "".join(outputs)
