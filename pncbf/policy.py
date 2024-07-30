import numpy as np

class Policy:
    def __init__(self):
        self.filter = None

    def __call__(self, state):
        # Hand-crafted policy
        base_command = 0.1 * (state.goal_pos - state.agent_pos)
        
        # Example filter
        filter_command = state.danger_radius ** 2 / (state.agent_pos - state.danger_pos) ** 2
        
        return base_command + filter_command