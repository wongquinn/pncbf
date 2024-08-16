import gurobipy as gp
import numpy as np
from gurobipy import GRB


class Solver:
    def __init__(self):
        # Storing a model and updating it is >2x better than creating a new one each time
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)  # Don't print diagnostics
        env.start()
        self.m = gp.Model(env=env)

        # Create a matrix variable for the optimal control
        self.u = self.m.addMVar(shape=2, vtype=GRB.CONTINUOUS, name="u")

    def alpha(self, b, k=1):
        """
        A class-kappa function. A higher coefficient makes the solution more conservative.

        Parameters:
        b (float): The CBF value.
        k (float): The coefficient.

        Returns:
        float: The output of alpha.
        """
        return k * b

    def solve(self, grad, f, g, b, u_nominal):
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
            # The objective is to minimize the difference from the nominal control
            obj = (self.u - u_nominal) @ (self.u - u_nominal)
            self.m.setObjective(obj, GRB.MINIMIZE)

            # The constraint is the descent condition
            direction = f.T + g @ self.u
            derivative = direction @ grad
            alpha_b = self.alpha(b)

            # Update constraint
            self.m.remove(self.m.getConstrs())
            self.m.addConstr(derivative <= -alpha_b)

            # Return the solution
            self.m.optimize()

            return self.u.X

        except Exception:
            # If something goes wrong, return a zero vector
            # print(f"Error {e}")
            return np.array([0, 0])
