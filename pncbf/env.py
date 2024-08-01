import numpy as np


class State:
    def __init__(self):
        "Default state"
        self.agent_pos = np.array([20.0, 20.0])
        self.agent_vel = np.array([0.0, 0.0])

        self.goal_pos = np.array([80.0, 80.0])
        self.goal_vel = np.array([0.0, 0.0])

        self.danger_pos = np.array([45.0, 55.0])
        self.danger_vel = np.array([0.0, 0.0])
        self.danger_radius = 15.0

        self.total_dim = 12  # All the positions and velocities

    def to_array(self):
        return np.concatenate(
            [
                self.agent_pos,
                self.agent_vel,
                self.goal_pos,
                self.goal_vel,
                self.danger_pos,
                self.danger_vel,
            ]
        )


class Environment:
    def __init__(self, args, policy):
        self.args = args
        self.policy = policy

        self.max_agent_vel = args.max_agent_vel
        self.world_dims = args.world_dims
        self.h_scale = 1.0

        self.state = State()
        self.info = {}

    def h(self, state):
        # normalized distance from danger zone
        d = np.linalg.norm(state.agent_pos - state.danger_pos) / state.danger_radius
        return self.h_scale * (1 - d**2) / (1 + d**2)

    def step(self):
        action = self.policy(self.state)
        self.state.agent_vel = self.process_action(action)

        self.state.agent_pos += self.state.agent_vel
        self.state.goal_pos += self.state.goal_vel
        self.state.danger_pos += self.state.danger_vel

        self.info = {"h": self.h(self.state)}

        return self.state, self.info

    def process_action(self, action):
        if np.linalg.norm(action) > self.max_agent_vel:
            action = action / np.linalg.norm(action) * self.max_agent_vel

        return action

    def reset(self):
        """Reset to default state"""
        self.state = State()
        self.info = {}
