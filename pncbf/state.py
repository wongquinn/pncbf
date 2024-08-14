import numpy as np
import torch


class State:
    """A class for storing the state of the environment."""

    def __init__(self):
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

    def to_tensor(self, device):
        return torch.tensor(self.to_array(), dtype=torch.float32).to(device)

    def set_to_default(self):
        """Set the state passed in to the default."""
        self.agent_pos = np.array([20.0, 20.0])
        self.agent_vel = np.array([0.0, 0.0])

        self.goal_pos = np.array([80.0, 80.0])
        self.goal_vel = np.array([0.0, 0.0])

        self.danger_pos = np.array([60.0, 60.0])
        self.danger_vel = np.array([0.0, 0.0])
        self.danger_radius = 15.0

        return self

    def randomize_agent(self, bounds):
        """Randomize the agent's position."""
        self.agent_pos = np.random.uniform(bounds[0], bounds[1], 2)

        return self

    def __repr__(self):
        return f"State(agent_pos={self.agent_pos}, agent_vel={self.agent_vel}, goal_pos={self.goal_pos}, goal_vel={self.goal_vel}, danger_pos={self.danger_pos}, danger_vel={self.danger_vel}, danger_radius={self.danger_radius})"

    def __iadd__(self, other):
        self.agent_pos += other.agent_pos
        self.agent_vel += other.agent_vel
        self.goal_pos += other.goal_pos
        self.goal_vel += other.goal_vel
        self.danger_pos += other.danger_pos
        self.danger_vel += other.danger_vel

        return self
