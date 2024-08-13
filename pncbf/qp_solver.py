import gurobipy as gp
from gurobipy import GRB

def alpha(b):
    """A class-kappa function."""
    return b ** 3

def solve(grad, f, g, b, u_nominal):
    """
    Solve the quadratic program.

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
        m = gp.Model("qp")
        
        # Create a matrix variable for the optimal control
        u = m.addMVar(shape=2, vtype=GRB.CONTINUOUS, name="u")
        
        # The objective is to minimize the difference from the nominal control
        obj = (u - u_nominal) @ (u - u_nominal)
        m.setObjective(obj, GRB.MINIMIZE)
        
        # The constraint is the descent condition
        direction = f + g @ u
        derivative = grad @ direction
        alpha_b = alpha(b)
        
        m.addConstr(derivative <= -alpha_b)
        
        # Return the solution
        m.optimize()
        return u.X
        
    except Exception as e:
        print(f"Error {e}")