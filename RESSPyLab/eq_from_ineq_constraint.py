"""@package eq_fom_ineq_constraints
Object to convert inequality to equality constraints through slack variables.
"""
import numpy as np
from rpl_constraint import RPLConstraint


class EqFromIneqConstraint(RPLConstraint):

    def __init__(self, constants, variables, constraints, constraint_gradients, constraint_hessians,
                 non_const_update_fun=None):
        """ Transforms inequality constraints to equality constraints through the use of slack variables.

        :param dict constants: Constant parameters in the constraint functions.
        :param dict variables: Non-constant parameters in the constraint functions.
        :param list constraints: All constraint functions to be imposed in the optimization procedure, each function
            returns either a float or an np.array if the function imposes multiple constraints.
        :param list constraint_gradients: Gradients of all the constraint functions, each function returns either an
            np.array of size (n, 1) for a single constraint or a list of length n_c for n_c constraints.
        :param list constraint_hessians: Hessians of all the constraint functions, each function returns either an
            np.array of size (n, n) for a single constraint or a list of length n_c for n_c constraints.
            g_i.
        :param function non_const_update_fun: Updates the variables that are used by the constraint functions.

        Notes:
            - See AugLagConstraint mother class for the assumptions on each of the input parameters.
            - All the constraints specified must be inequality constraints of the form g_i(x) <= 0, i=1,2,...,m
            - Since all the constraints are inequality constraints, the number of slack variables added is equal to the
                number of constraints (num slacks = m)
        """
        RPLConstraint.__init__(self, constants, variables, constraints, constraint_gradients, constraint_hessians,
                               non_const_update_fun)
        self.num_slacks = None
        self.num_primals = None
        self._set_n_slacks = False
        return

    def set_n_slacks_primals(self, x_0):
        if self._set_n_slacks is False:
            self.num_slacks = len(RPLConstraint.get_g(self, x_0))
            self.num_primals = len(x_0)
        else:
            raise RuntimeError('Already set the number of slacks.')
        return

    def get_g(self, x):
        """ Returns the value of each constraint for all the equality constraints.

        :param np.array x: (n+m, 1) Primal variables, including the slack variables.
        :return np.array: (m, 1) Equality constraint values, m = number of constraints.
        """
        primals = x[:self.num_primals]
        slacks = x[self.num_primals:]
        all_g = RPLConstraint.get_g(self, primals)
        for i, gi in enumerate(all_g):
            all_g[i] = gi + slacks[i] ** 2
        return all_g

    def get_gradient(self, x):
        """ Returns the gradients for all of the equality constraints.

        :param np.array x: (n+m, 1) Primal variables, including the slack variables.
        :return list: Collection of gradients of g wrt x, each element is an np.array shape=(n+m, 1)
        """
        primals = x[:self.num_primals]
        slacks = x[self.num_primals:]
        all_grad = RPLConstraint.get_gradient(self, primals.reshape(-1))
        for i, dgi in enumerate(all_grad):
            slack_vec = np.zeros((self.num_slacks, 1))
            slack_vec[i] = 2. * slacks[i]
            all_grad[i] = np.row_stack((dgi, slack_vec))
        return all_grad

    def get_hessian(self, x):
        """ Returns the Hessians for all the equality constraints.

        :param np.array x: (n+m, 1) Primal variables, including the slack variables.
        :return list: Collection of Hessians of g wrt x, each is an np.array shape=(n+m, n+m)
        """
        total_size = len(x)
        primals = x[:self.num_primals]
        all_hess = RPLConstraint.get_hessian(self, primals)
        for i, hgi in enumerate(all_hess):
            big_mat = np.zeros((total_size, total_size)) * 1.0
            big_mat[:self.num_primals, :self.num_primals] = hgi
            big_mat[self.num_primals + i, self.num_primals + i] = 2.
            all_hess[i] = big_mat
        return all_hess

    def get_primals_slacks(self, x):
        """ Separates the primal and slack variables. """
        primals = x[:self.num_primals]
        slacks = x[self.num_primals:]
        return [primals, slacks]

    def append_0_slacks(self, x):
        """ Returns x including the slack variables with all the slack variables set to zero.

        :param np.array x: (n, 1) Primal variables, not including slack variables.
        :return np.array: (n+m, 1) Primal variables including slack variables.
        """
        return np.append(x, np.zeros((self.num_slacks, 1))).reshape(-1, 1)

    def append_0_slacks_matrix(self, h):
        """ Returns matrix including slack variables set to zero, h is a square matrix. """
        n, _ = np.shape(h)
        big_mat = np.zeros((n + self.num_slacks, n + self.num_slacks))
        big_mat[:n, :n] = h
        return big_mat

    def calc_initial_x_slacks(self, x):
        slacks = np.zeros((self.num_slacks, 1)).reshape(-1, 1)
        all_g = RPLConstraint.get_g(self, x)
        for i, gi in enumerate(all_g):
            if gi < 0:
                slacks[i] = np.sqrt(-1. * gi)
            else:
                slacks[i] = np.sqrt(1. * gi)
        return np.append(x, slacks).reshape(-1, 1)
