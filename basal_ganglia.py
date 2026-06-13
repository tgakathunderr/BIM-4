import numpy as np
from bim4.config import N, BABBLE_THRESHOLD, BABBLE_RATE

class BasalGanglia:
    """
    Action selection and reinforcement learning engine.
    Maps Cortical SDRs to Expected Rewards for Motor Actions (bytes).
    """
    def __init__(self, action_space_size=256):
        self.action_space_size = action_space_size
        
        # Matrix mapping: [motor_action, cortical_neuron_index] -> expected_reward
        self.weights = np.zeros((self.action_space_size, N), dtype=np.float32)
        
        self.rng = np.random.RandomState(42)

    def score_actions(self, cortical_sdr: np.ndarray) -> np.ndarray:
        """
        Calculates the expected reward for all possible actions given the current cortical state.
        cortical_sdr: array of active indices.
        """
        # Sum the weights for the active cortical neurons for each action
        # This is a fast sparse dot product.
        if len(cortical_sdr) == 0:
            return np.zeros(self.action_space_size, dtype=np.float32)
            
        expected_rewards = np.sum(self.weights[:, cortical_sdr], axis=1)
        return expected_rewards

    def select_action(self, expected_rewards: np.ndarray) -> tuple[int, bool]:
        """
        Selects an action based on expected rewards, applying babbling (exploration) if needed.
        Returns: (chosen_byte, is_babbling)
        """
        best_action = int(np.argmax(expected_rewards))
        max_reward = expected_rewards[best_action]
        
        is_babbling = False
        
        # If confidence is too low, or by random chance, babble.
        if max_reward < BABBLE_THRESHOLD or self.rng.random() < BABBLE_RATE:
            best_action = self.rng.randint(0, self.action_space_size)
            is_babbling = True
            
        return best_action, is_babbling

    def learn(self, cortical_sdr: np.ndarray, chosen_action: int, dopamine: float):
        """
        Updates the action weights based on the dopamine reward signal.
        dopamine: ranges from -1.0 to 1.0
        """
        if len(cortical_sdr) == 0:
            return
            
        # Hebbian RL: Strengthen the connection between the active cortical state
        # and the chosen action, scaled by dopamine.
        # If dopamine is negative (pain), this decreases the weight.
        
        learning_rate = 0.1
        weight_update = dopamine * learning_rate
        
        self.weights[chosen_action, cortical_sdr] += weight_update
        
        # Clip weights to prevent them from exploding infinitely
        np.clip(self.weights, -10.0, 10.0, out=self.weights)
