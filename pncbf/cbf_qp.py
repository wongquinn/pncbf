import gurobipy


class CbfQp():
    ''' This class represents the CBF quadratic program described by
        equation (5) of http://arxiv.org/abs/2310.15478. '''
    
    def __init__(self, shape_u):
        self.shape_u = shape_u

        self.model = gurobipy.Model("patrolling")
        self.add_variables()
        self.add_constraints()
        self.add_objective()


    def add_variables(self):
        ''' Initializes variables of the model. '''

        # Create a matrix variable for the control output.
        self.u = self.model.addMVar(shape=self.shape_u, vtype=gurobipy.GRB.CONTINUOUS, name="u")
    

    def add_constraints(self):
        ''' Initializes constraints of the model. '''

        # We set up a basic constraint here with u <= 0. This is a placeholder.
        # When we actually solve, we will set the LHS coefficient and RHS value.
        self.constr = self.model.addConstr(self.u <= 0)


    def add_objective(self):
        ''' Initializes the objective function of the model. '''

        # The objective function is to minimize the L2 norm of the difference
        # between the current and nominal visit times.
        # No point in setting it here as we will update before solving.
        pass
        # self.model.setObjective(gurobipy.norm(self.u - 0.0, 2.0), gurobipy.GRB.MINIMIZE)


    def reset(self):
        ''' Resets the model to an unsolved state. '''

        self.model.reset()
    

    def solve(self, u_nominal, lie_deriv_f_bx, lie_deriv_g_bx, alpha_bx):
        ''' This function solves the CBF QP given the nominal control input,
            the Lie derivatives, and the RHS.
            We do not reset the model here so as to allow warm-starting. '''
        
        # Update the objective function given the nominal control input.
        self.model.setObjective(gurobipy.norm(self.u - u_nominal, 2.0), gurobipy.GRB.MINIMIZE)

        # Update the constraint given the Lie derivatives and RHS.
        self.constr.setAttr(gurobipy.GRB.Attr.RHS, -alpha_bx - lie_deriv_f_bx)
        self.model.chgCoeff(self.constr, self.u, lie_deriv_g_bx)

        # Solve the model.
        self.model.update()
        self.model.optimize()

        # Return the optimal control input.
        return self.u.X

# Test the CBF QP.
import numpy as np
cbf_qp = CbfQp((3,))
u_nominal = np.array([0.1, 0.2, 0.3])
lie_deriv_f_bx = np.array([0.1, 0.2, 0.3])
lie_deriv_g_bx = np.array([0.1, 0.2, 0.3])
alpha_bx = np.array([0.1, 0.2, 0.3])
u = cbf_qp.solve(u_nominal, lie_deriv_f_bx, lie_deriv_g_bx, alpha_bx)
print(u)