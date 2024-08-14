class Policy:
    """A class that holds the nominal policy and the filter."""
    
    def __init__(self, args, filter):
        self.args = args
        self.filter = filter

    def __call__(self, state):
        # Hand-crafted policy
        nominal_action = 0.1 * (state.goal_pos - state.agent_pos)

        modified_action = self.filter(state, nominal_action)
        return modified_action
