class Hippocampus:
    """
    Episodic memory and sleep consolidation.
    Stores recent experiences and replays them to train the Cortex and Basal Ganglia offline.
    """
    def __init__(self, capacity=2000):
        self.capacity = capacity
        # We store tuples of (sensory_sdr, motor_action, dopamine_level, ach_level)
        self.buffer = [] 
        self.cursor = 0
        
    def store(self, sdr, action, dopamine, ach):
        """ Stores a single timestep episode. """
        if len(self.buffer) < self.capacity:
            self.buffer.append((sdr, action, dopamine, ach))
        else:
            self.buffer[self.cursor] = (sdr, action, dopamine, ach)
            self.cursor = (self.cursor + 1) % self.capacity
            
    def get_dream_sequence(self):
        """
        Returns the stored episodes for offline replay in chronological order.
        """
        if len(self.buffer) == 0:
            return []
            
        # Reconstruct sequence in chronological order
        ordered = self.buffer[self.cursor:] + self.buffer[:self.cursor]
        return ordered
        
    def clear(self):
        """ Clears memory after a successful sleep cycle. """
        self.buffer = []
        self.cursor = 0
