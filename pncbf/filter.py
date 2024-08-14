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
    """A filter that finds a minimally-different safe action using QP."""
    def __init__(self, args, model=None):
        self.args = args
        self.ncbf_model = model

    def __call__(self, state, nominal_action):
        state_tensor = state.to_tensor(self.args.device)
        grad = calculate_gradient(self.ncbf_model, state_tensor)
        f, g = get_affine_dynamics(state)
        b = self.ncbf_model(state_tensor).item()

        return solve(grad, f, g, b, nominal_action)


class NeuralFilter:
    def __init__(self, args, model=None):
        self.args = args
        self.qp_model = model

    def __call__(self, state, base_action):
        input = torch.tensor(state.to_array(), dtype=torch.float32).to(self.args.device)
        return self.qp_model(input).cpu().detach().numpy()
