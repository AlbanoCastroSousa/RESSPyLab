"""@package rpl_constraint
Defines the RPLConstraint object to apply constraints to the RESSPyLab minimization problems.
"""
import numpy as np


class RPLConstraint:
    def __init__(self, constants, variables, constraints, constraint_gradients, constraint_hessians,
                 non_const_update_fun=None):
        """ Container for the constraints, gradients, and Hessians specified in the RESSPyLab optimization models.

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

        - All constraints are assumed to be inequalities of the form g_i(x) <= 0
        - All of the functions passed into this class must be coherent with each other in terms of the constant and
        non-constant parameters, as well as the model parameters (x) that each function uses.
        - Each of the functions in the lists are functions of: x, c_vars, nc_vars.
        - The order of the gradients and Hessians in the lists should match the order of the constraint functions.
        - For the gradients and Hessians, if one function imposes multiple constraints the lists should contain
            vectors or size (n, 1) or (n, n).
        - If specified, one function should update all the non-constant parameters.
        """
        self.c_vars = constants
        self.nc_vars = variables
        self.list_g = constraints
        self.list_dg = constraint_gradients
        self.list_hg = constraint_hessians
        self.nc_var_update = non_const_update_fun

    def get_g(self, x):
        """ Returns the value of each constraint for all the constraint functions.

        :param np.array x: Model parameters in the constraint functions.
        :return np.array: (m, 1) Constraint values, m = number of constraints, each element is a float.
        """
        all_g = np.array([])
        for g_fun in self.list_g:
            g = g_fun(x, self.c_vars, self.nc_vars)
            if type(g) is list:
                for gi in g:
                    all_g = np.append(all_g, gi)
            else:
                all_g = np.append(all_g, g)

        return all_g

    def get_gradient(self, x):
        """ Returns the gradients for all the constraint functions.

        :param np.array x: Model parameters in the constraint functions
        :return list: Collection of gradients of g wrt x, each element is an np.array (n, 1), n = len(x)
        """
        all_grad = []
        for grad_fun in self.list_dg:
            dg = grad_fun(x, self.c_vars, self.nc_vars)
            if type(dg) is list:
                for dgi in dg:
                    all_grad.append(dgi)
            else:
                all_grad.append(dg)

        return all_grad

    def get_hessian(self, x):
        """ Returns the sum of the Hessians for all the constraint functions.

        :param np.array x: Model parameters in the constraint functions
        :return list: Collection of Hessians of g wrt x, each is an np.array (n, n), n = len(x)
        """
        all_hess = []
        for hess_fun in self.list_hg:
            hg = hess_fun(x, self.c_vars, self.nc_vars)
            if type(hg) is list:
                for hgi in hg:
                    all_hess.append(hgi)
            else:
                all_hess.append(hg)

        return all_hess

    def update_variables(self, x):
        """ Updates the non-constant variables in the constraint functions.

        :param np.array x: Model parameters in the constraint functions
        """
        if self.nc_var_update is None:
            return
        else:
            self.nc_vars = self.nc_var_update(x, self.c_vars, self.nc_vars)
            return
