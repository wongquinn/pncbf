import numpy as np
import torch


class NullFilter:
    def __init__(self, args):
        self.args = args

    def __call__(self, state, base_action):
        return base_action


class HandmadeFilter:
    def __init__(self, args):
        self.args = args
        self.k = 1000

    def __call__(self, state, base_action):
        vec = state.agent_pos - state.danger_pos
        filter_action = self.k * vec / np.linalg.norm(vec) ** 3
        
        return base_action + filter_action


class QPFilter:
    def __init__(self, args):
        self.args = args
        self.ncbf = None

    def __call__(self, state, base_action):
        pass


class NeuralFilter:
    def __init__(self, args, model=None):
        self.args = args
        self.qp_model = model

    def __call__(self, state, base_action):
        input = torch.tensor(state.to_array(), dtype=torch.float32).to(self.args.device)
        return self.qp_model(input).cpu().detach().numpy()
