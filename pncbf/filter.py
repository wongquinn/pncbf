class NullFilter:
    def __init__(self, args):
        self.args = args

    def __call__(self, state, base_command):
        return base_command


class HandmadeFilter:
    def __init__(self, args):
        self.args = args

    def __call__(self, state, base_command):
        filter_command = (
            state.danger_radius**2 / (state.agent_pos - state.danger_pos) ** 2
        )

        return base_command + filter_command
