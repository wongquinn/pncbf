import numpy as np
from pncbf.state import State


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
        raw_action = self.policy(self.state)
        action = self.process_action(raw_action)
        self.state.agent_vel = action

        self.state.agent_pos += self.state.agent_vel
        self.state.goal_pos += self.state.goal_vel
        self.state.danger_pos += self.state.danger_vel

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

    def reset(self):
        """Reset to default state"""
        self.state = State()
        self.state.set_to_default()
        self.state.randomize_agent(self.world_dims)

        self.info = {}
