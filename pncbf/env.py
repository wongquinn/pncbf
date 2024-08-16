import numpy as np
from pncbf.state import State


class Environment:
    """A class that holds the environment state and dynamics."""

    def __init__(self, args, policy):
        self.args = args
        self.policy = policy

        self.max_agent_vel = args.max_agent_vel
        self.world_dims = args.world_dims
        self.h_scale = 1.0

        self.state = State()
        self.info = {}

    def h(self, state):
        """Normalized distance from danger zone."""
        d = np.linalg.norm(state.agent_pos - state.danger_pos) / state.danger_radius
        return self.h_scale * (1 - d**2) / (1 + d**2)

    def step(self):
        """State transition function."""
        raw_action = self.policy(self.state)
        action = self.process_action(raw_action)
        self.state += self.state_derivative(self.state, action)

        self.info = {
            "h": self.h(self.state),
            "raw_action": raw_action,
            "action": action,
        }

        return self.state, self.info

    def process_action(self, action):
        """Process the action such as by applying actuation limits"""
        if np.linalg.norm(action) > self.max_agent_vel:
            action = action / np.linalg.norm(action) * self.max_agent_vel

        return action

    def state_derivative(self, state, action):
        """Calculate x_dot given the current state x and an action u"""
        x_dot = State()

        x_dot.agent_pos = action
        x_dot.agent_vel = np.zeros_like(state.goal_vel)
        x_dot.goal_pos = state.goal_vel
        x_dot.goal_vel = np.zeros_like(state.goal_vel)
        x_dot.danger_pos = state.danger_vel
        x_dot.danger_vel = np.zeros_like(state.danger_vel)

        return x_dot

    def reset(self, agent_bounds=None):
        """Reset to default state"""
        self.state = State()
        self.state.set_to_default()
        if agent_bounds is None:
            self.state.randomize_agent(self.world_dims)
        else:
            self.state.randomize_agent(agent_bounds)

        self.info = {}


def get_affine_dynamics(state):
    """Get the f and g matrices for a given state."""
    f = np.zeros((12, 1))
    g = np.zeros((12, 2))

    # Derivative of the goal and agent positions are their velocities
    f[4:6, 0] = state.goal_vel
    f[8:10, 0] = state.danger_vel

    # Derivative of the agent's position is the action
    g[0:2, 0:2] = np.eye(2)

    # The agent's velocity changes by the action minus the original value
    f[2:4, 0] = -state.agent_vel
    g[2:4, 0:2] = np.eye(2)

    return f, g
