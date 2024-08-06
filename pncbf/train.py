import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
import torch.nn as nn

from pncbf.env import Environment, State
from pncbf.filter import HandmadeFilter, NullFilter
from pncbf.nn import MLP
from pncbf.policy import Policy
from pncbf.renderer import NotebookRenderer
from tqdm import tqdm


def render_env(r, state=None, info=None):
    r.set_data("agent", *state.agent_pos, fmt="ob", label="agent")
    r.set_data("goal", *state.goal_pos, fmt="og", label="goal")

    danger_patch = plt.Circle(
        state.danger_pos,
        state.danger_radius,
        color="r",
        alpha=0.5,
        label="danger",
    )
    r.set_patch("danger", danger_patch)
    r.draw()

    if state is not None:
        print("State array:")
        print(state.to_array())
    if info is not None:
        print(f"h: {info['h']:.4f}")


def collect_tuples(args):
    np.set_printoptions(precision=2, suppress=True)

    rollouts = args.rollouts
    steps = args.steps_per_rollout

    filter = NullFilter(args)
    policy = Policy(args, filter)
    env = Environment(args, policy)
    r = NotebookRenderer(xlim=args.world_dims, enable_legend=True)

    state_dim = env.state.total_dim

    # Each row is the current state, the max future h, and the final state
    rollout_data = np.zeros((rollouts, steps, state_dim + 1 + state_dim))
    for i in range(rollouts):
        env.reset()

        for j in range(steps):
            state, info = env.step()

            # The first part of the row is the state at each step
            rollout_data[i, j, :state_dim] = state.to_array()
            rollout_data[i, j:, state_dim] = info["h"]

            if args.render:
                render_env(r, state, info)

        # Set the h column to the max future h
        for j in range(steps - 2, -1, -1):
            rollout_data[i, j, state_dim] = max(
                rollout_data[i, j, state_dim],
                rollout_data[i, j + 1, state_dim],
            )

        # The last part of the row is the final state
        rollout_data[i, :, state_dim + 1 :] = state.to_array()

    return rollout_data


def train_ncbf(args, rollout_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = (rollout_data.shape[-1] - 1) // 2

    # Extract the input (current state) and target (max future h) from the rollout data
    inputs = rollout_data[:, :, :state_dim]
    targets = rollout_data[:, :, state_dim]

    # Reshape the inputs and targets to match the MLP input shape
    inputs = np.reshape(inputs, (-1, state_dim))
    targets = np.reshape(targets, (-1, 1))

    # Convert inputs and targets to tensors
    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
    targets = torch.tensor(targets, dtype=torch.float32).to(device)

    # Create an MLP model
    model = MLP(input_dim=state_dim, hidden_dims=[128], output_dim=1).to(device)

    # Define the loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in (pbar := tqdm(range(args.num_epochs))):
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"[{epoch+1}/{args.num_epochs}] | Loss: {loss.item():.4f}")

    # Return the trained model
    return model


def render_ncbf(args, model):
    r = NotebookRenderer(xlim=args.world_dims, enable_legend=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = args.contour_grid_size
    levels = args.contour_levels

    # Create grid values
    x = np.linspace(args.world_dims[0], args.world_dims[1], size)
    y = np.linspace(args.world_dims[0], args.world_dims[1], size)
    X, Y = np.meshgrid(x, y)

    # Stack the state array vertically and set the position of the agent
    inputs = np.tile(State().to_array(), (size * size, 1))
    inputs[:, 0] = X.flatten()
    inputs[:, 1] = Y.flatten()

    # Convert inputs to a tensor
    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

    # Evaluate the model over every point in the grid
    outputs = model(inputs)

    # Reshape the output tensor to match the grid shape
    Z = outputs.reshape(size, size).detach().cpu().numpy()

    # Set h=0 to be the contour level 0
    norm = TwoSlopeNorm(vmin=Z.min(), vcenter=0.0, vmax=Z.max())

    contour = plt.gca().contourf(X, Y, Z, levels=levels, norm=norm, cmap="RdBu_r")
    plt.gcf().colorbar(contour, label="h")

    # Render the environment on top of the contour plot.
    render_env(r, State())
