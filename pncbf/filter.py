import numpy as np
import torch
from pncbf.env import get_affine_dynamics
from pncbf.utils import calculate_gradient
from pncbf.qp_solver import Solver


class NullFilter:
    """A filter that does not modify the nominal action."""

    def __init__(self, args):
        self.args = args

    def __call__(self, state, nominal_action):
        return nominal_action


class HandmadeFilter:
    """A filter that repels the agent from the danger zone."""

    def __init__(self, args):
        self.args = args
        self.k = 1000

    def __call__(self, state, nominal_action):
        vec = state.agent_pos - state.danger_pos
        filter_action = self.k * vec / np.linalg.norm(vec) ** 3

        return nominal_action + filter_action


class QPFilter:
    """A filter that finds a minimally-different safe action using QP."""

    def __init__(self, args, model=None):
        self.args = args
        self.ncbf_model = model
        self.solver = Solver()

    def __call__(self, state, nominal_action):
        state_tensor = state.to_tensor(self.args.device)
        grad = calculate_gradient(self.ncbf_model, state_tensor)
        f, g = get_affine_dynamics(state)
        b = self.ncbf_model(state_tensor).item()

        return self.solver.solve(grad, f, g, b, nominal_action)


class NeuralFilter:
    """A filter that passes the nominal action through an NN."""

    def __init__(self, args, model=None):
        self.args = args
        self.qp_model = model

    def __call__(self, state, nominal_action):
        input = torch.tensor(state.to_array(), dtype=torch.float32).to(self.args.device)
        return self.qp_model(input).cpu().detach().numpy()
