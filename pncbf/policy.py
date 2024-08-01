class Policy:
    def __init__(self, args, filter):
        self.args = args
        self.filter = filter

    def __call__(self, state):
        # Hand-crafted policy
        base_command = 0.1 * (state.goal_pos - state.agent_pos)

        modified_command = self.filter(state, base_command)
        return modified_command
