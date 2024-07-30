import numpy as np


class State:
    def __init__(self):
        self.agent_pos = np.array([0.0, 0.0])
        self.agent_vel = np.array([0.0, 0.0])

        self.goal_pos = np.array([0.0, 0.0])
        self.goal_vel = np.array([0.0, 0.0])

        self.danger_pos = np.array([0.0, 0.0])
        self.danger_vel = np.array([0.0, 0.0])
        self.danger_radius = 0.0


class Environment:
    def __init__(self, args, policy):
        self.args = args
        self.policy = policy

        self.max_agent_vel = args.max_agent_vel
        self.world_size = args.world_size

        self.state = State()
        self.info = {}

        self.reset()

    def step(self):
        action = self.policy(self.state)
        self.state.agent_vel = self.process_action(action)

        self.state.agent_pos += self.state.agent_vel
        self.state.goal_pos += self.state.goal_vel
        self.state.danger_pos += self.state.danger_vel

        return self.state, self.info

    def process_action(self, action):
        if np.linalg.norm(action) > self.max_agent_vel:
            action = action / np.linalg.norm(action) * self.max_agent_vel

        return action

    def reset(self):
        """Reset to default state"""
        self.state.agent_pos = np.array([20.0, 20.0])
        self.state.agent_vel = np.array([0.0, 0.0])

        self.state.goal_pos = np.array([80.0, 80.0])
        self.state.goal_vel = np.array([0.0, 0.0])

        self.state.danger_pos = np.array([45.0, 55.0])
        self.state.danger_vel = np.array([0.0, 0.0])
        self.state.danger_radius = 15.0

        self.info = {}
