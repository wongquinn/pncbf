import gurobipy as gp
import numpy as np
from gurobipy import GRB


def alpha(b, k=1):
    """
    A class-kappa function. A higher coefficient makes the solution more conservative.

    Parameters:
    b (float): The CBF value.
    k (float): The coefficient.

    Returns:
    float: The output of alpha.
    """
    return k * b


def solve(grad, f, g, b, u_nominal):
    """
    Solve the quadratic program for a safe solution.

    Parameters:
    - grad (numpy.ndarray): The gradient of the CBF at the target state.
    - f (numpy.ndarray): Affine dynamics matrix.
    - g (numpy.ndarray): Affine dynamics matrix.
    - alpha_b (float): alpha(cbf(x)).
    - u_nominal (numpy.ndarray): The nominal control input.

    Returns:
    - solution (numpy.ndarray): The solution to the quadratic program.
    """
    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0) # Don't print diagnostics
            env.start()
            with gp.Model(env=env) as m:
                # Create a matrix variable for the optimal control
                u = m.addMVar(shape=2, vtype=GRB.CONTINUOUS, name="u")

                # The objective is to minimize the difference from the nominal control
                obj = (u - u_nominal) @ (u - u_nominal)
                m.setObjective(obj, GRB.MINIMIZE)

                # The constraint is the descent condition
                direction = f.T + g @ u
                derivative = direction @ grad
                alpha_b = alpha(b)
                m.addConstr(derivative <= -alpha_b)

                # Return the solution
                m.optimize()

                return u.X

    except Exception:
        # If something goes wrong, return a zero vector
        # print(f"Error {e}")
        return np.array([0, 0])
