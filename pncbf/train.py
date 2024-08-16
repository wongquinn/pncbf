import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from pncbf.env import Environment
from pncbf.filter import NeuralFilter, QPFilter
from pncbf.nn import MLP
from pncbf.policy import Policy
from pncbf.renderer import NotebookRenderer
from pncbf.state import State


def render_env(r, state, info=None):
    """
    Render a frame of the environment with the agent, goal, and danger positions.

    Args:
        r (NotebookRenderer): The renderer object.
        state (State, optional): The current state of the environment. Defaults to None.
        info (dict, optional): Additional information about the environment. Defaults to None.
    """
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

    if info is not None:
        print("State array:")
        print(state.to_array())
        print(f"h: {info['h']:.4f}")
        print(f"action: {info['action']}")


def collect_ncbf(args):
    """
    Collect tuples of states, max future h, and final states from the environment.

    Args:
        args: The arguments for collecting tuples.

    Returns:
        np.ndarray: The collected tuples.
    """
    policy = Policy(args, args.filter(args))
    env = Environment(args, policy)
    r = NotebookRenderer(xlim=args.world_dims, enable_legend=True)
    state_dim = env.state.total_dim

    # Each row is the current state, the max future h, and the final state
    rollout_data = np.zeros(
        (
            args.rollouts,
            args.steps_per_rollout,
            state_dim + 1 + state_dim,
        )
    )
    for i in tqdm(range(args.rollouts)):
        # Prepare for rollout
        env.reset()

        for j in range(args.steps_per_rollout):
            state, info = env.step()

            # The first part of the row is the state at each step
            rollout_data[i, j, :state_dim] = state.to_array()
            rollout_data[i, j:, state_dim] = info["h"]

            if args.render:
                render_env(r, state, info)

        # Set the h column to the max future h
        for j in range(args.steps_per_rollout - 2, -1, -1):
            rollout_data[i, j, state_dim] = max(
                rollout_data[i, j, state_dim],
                rollout_data[i, j + 1, state_dim],
            )

        # The last part of the row is the final state
        rollout_data[i, :, state_dim + 1 :] = state.to_array()

    return rollout_data


def train_ncbf(args, rollout_data):
    """
    Train a NCBF (Neural Control Barrier Function) model which predicts the max future safety violation (h).

    Args:
        args: The arguments for training the model.
        rollout_data (np.ndarray): The collected tuples.

    Returns:
        MLP: The trained NCBF model.
    """

    state_dim = (rollout_data.shape[-1] - 1) // 2

    # Extract the input (current state) and target (max future h) from the rollout data
    inputs = rollout_data[:, :, :state_dim]
    targets = rollout_data[:, :, state_dim]

    # Reshape the inputs and targets to match the MLP input shape
    inputs = np.reshape(inputs, (-1, state_dim))
    targets = np.reshape(targets, (-1, 1))

    # Convert inputs and targets to tensors
    inputs = torch.tensor(inputs, dtype=torch.float32).to(args.device)
    targets = torch.tensor(targets, dtype=torch.float32).to(args.device)

    # Create an MLP model
    model = MLP(
        input_dim=state_dim,
        hidden_dims=args.hidden_dims,
        output_dim=1,
    ).to(args.device)

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


def test_ncbf(args, ncbf_model):
    """
    Render the NCBF model as a contour map with the default state.

    Args:
        args: The arguments for rendering the model.
        model (MLP): The trained NCBF model.
    """
    r = NotebookRenderer(xlim=args.world_dims, enable_legend=True)

    size = args.contour_grid_size
    levels = args.contour_levels

    # Create grid values
    x = np.linspace(args.world_dims[0], args.world_dims[1], size)
    y = np.linspace(args.world_dims[0], args.world_dims[1], size)
    X, Y = np.meshgrid(x, y)

    # Stack the state array vertically and set the position of the agent
    default_state = State().set_to_default()
    inputs = np.tile(default_state.to_array(), (size * size, 1))
    inputs[:, 0] = X.flatten()
    inputs[:, 1] = Y.flatten()

    # Convert inputs to a tensor
    inputs = torch.tensor(inputs, dtype=torch.float32).to(args.device)

    # Evaluate the model over every point in the grid
    outputs = ncbf_model(inputs)

    # Reshape the output tensor to match the grid shape
    Z = outputs.reshape(size, size).detach().cpu().numpy()

    contour = plt.gca().contourf(X, Y, Z, levels, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.gcf().colorbar(contour, label="h")

    # Render the environment on top of the contour plot.
    render_env(r, default_state)


def collect_qp(args, ncbf_model):
    """
    Collect data about the quadratic programming solution that minimally modifies the nominal policy while maintaining the descent condition.

    Args:
        args: The arguments for collecting QP data.
        model (MLP): The trained NCBF model.

    Returns:
        np.ndarray: The collected QP data.
    """
    policy = Policy(args, QPFilter(args, ncbf_model))
    # policy = Policy(args, HandmadeFilter(args))
    env = Environment(args, policy)
    r = NotebookRenderer(xlim=args.world_dims, enable_legend=True)
    state_dim = env.state.total_dim

    # Each row is the current state and the optimal QP solution (a vector in R^2)
    rollout_data = np.zeros(
        (
            args.rollouts,
            args.steps_per_rollout,
            state_dim + 2,
        )
    )
    for i in tqdm(range(args.rollouts)):
        # Prepare for rollout
        env.reset()

        for j in range(args.steps_per_rollout):
            state, info = env.step()

            # The first part of the row is the state at each step
            rollout_data[i, j, :state_dim] = state.to_array()
            rollout_data[i, j:, state_dim:] = info["action"]

            if args.render:
                render_env(r, state, info)

    return rollout_data


def train_qp(args, rollout_data):
    """
    Train a model that predicts the optimal QP solution.

    Args:
        args: The arguments for training the QP model.
        qp_data: The collected QP data.

    Returns:
        MLP: The trained QP model.
    """

    state_dim = rollout_data.shape[-1] - 2

    # Extract the input (current state) and target (max future h) from the rollout data
    inputs = rollout_data[:, :, :state_dim]
    targets = rollout_data[:, :, state_dim:]

    # Reshape the inputs and targets to match the MLP input shape
    inputs = np.reshape(inputs, (-1, state_dim))
    targets = np.reshape(targets, (-1, 2))

    # Convert inputs and targets to tensors
    inputs = torch.tensor(inputs, dtype=torch.float32).to(args.device)
    targets = torch.tensor(targets, dtype=torch.float32).to(args.device)

    # Create an MLP model
    model = MLP(
        input_dim=state_dim,
        hidden_dims=args.hidden_dims,
        output_dim=2,
    ).to(args.device)

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


def test_qp(args, qp_model):
    """
    Test the QP model on the default scenario.

    Args:
        args: The arguments for testing the QP model.
        model (MLP): The trained NCBF model.
        qp_model: The trained QP model.
    """
    policy = Policy(args, NeuralFilter(args, qp_model))
    env = Environment(args, policy)
    r = NotebookRenderer(xlim=args.world_dims, enable_legend=True)

    for i in range(args.rollouts):
        # Prepare for rollout
        env.reset([0, 60])  # Test positions may intersect the danger zone

        for j in range(args.steps_per_rollout):
            state, info = env.step()

            render_env(r, state, info)
