"""@package sqp_linsearch
Abstract class for RESSPyLab SQP solvers.
"""
import numpy as np


class SqpSolver:

    def __init__(self, objective_function, constraint, dumper=None):
        """ Abstract class to define SQP solvers given the objective function to minimize and the constraints to apply.

        :param MatModelErrorNda objective_function: Provides the objective function, and gradient / Hessian of it.
        :param AugLagConstraint constraint: Provides the constraints, and gradients / Hessians of them.
        :param Dumper dumper: Used to output information to the screen and/or to a file.

        The problem to be solved is to minimize an objective function, f(x), subjected to some constraints. Formally,

        minimize f(x)
            x
        subjected to g_i(x) <= 0

        where g_i(x) are nonlinear (possibly linear) constraint functions, i = 1, 2, ..., m are the number of
        constraints. The SQP method solves this problem by linearizing the constraints and solving a quadratic model of
        f(x) at each iteration k:

        minimize q(x) = grad[f(x_k)]^T . x_k  +  1/2 x_k^T . H_k . x_k
            x
        subjected to grad[g(x_k)]^T . x_k + g(x_k) <= 0,

        where H_k is an approximation of the Hessian of f(x_k). For additional details see the references below.

        References:
            [1] Bierlaire (2015) "Optimization: Principles and Algorithms"
            [2] Nocedal and Wright (2006) "Numerical optimization"
        """
        self.total_iterations = 0
        self.maximum_iterations = 3000
        self.precision = np.sqrt(np.finfo(float).eps)
        self.constraint = constraint
        self.objective_fun = objective_function
        if dumper is None:
            self.use_dumper = False
        else:
            self.use_dumper = True
            self.dumper = dumper

        # Used to let the all parts of the solver be aware of the active constraints
        self.active_constraints_index = 0
        self.active_constraints_set = False

        # Used for exit information
        self.convergence_reached_tag = 1
        self.maximum_iterations_reached_tag = 2
        self.unknown_exit = 99
        return

    def reset_solver(self):
        """ Sets the internal iteration count and active constraints to their starting values. """
        self.total_iterations = 0
        self.active_constraints_index = 0
        self.active_constraints_set = False
        return

    def set_maximum_iterations(self, n):
        """ Sets the maximum iterations to n. """
        self.maximum_iterations = n
        return

    def set_tolerance(self, tol):
        """ Sets the precision to tol. """
        self.precision = tol
        return

    def merit_fun(self, x, c):
        """ Merit function used to ensure global convergence. """
        raise Exception("Not implemented in {0}".format(self))

    def globalized_sqp(self, x_0, dual_x_0):
        """ Pure virtual method for solving the globalized SQP problem. """
        raise Exception("Not implemented in {0}".format(self))

    def hess_xx_lagrangian(self, x, hess_f, dual_x):
        """ Returns the Hessian of the Lagrangian only with respect to xx.

        :param np.array x: (n, 1) Primal variables.
        :param np.array hess_f: (n, n) Hessian of the objective function.
        :param np.array dual_x: (m, 1) Dual variables.
        :return np.array: (n, n) Hessian of the Lagrangian wrt. xx.

        Note returned Hessian is only the "upper left corner" of the Hessian of the problem.
        """
        hess = hess_f
        constraint_hessians = self.constraint.get_hessian(x)
        for i, hi in enumerate(constraint_hessians):
            hess = hess + dual_x[i] * hi
        return hess

    def grad_lagrangian(self, x, grad_f, dual_x, constraint_array, active_constraints=None):
        """ Returns the gradient of the problem.

        :param np.array x: (n, 1) Primal variables.
        :param np.array grad_f: (n, 1) Gradient of the objective function.
        :param np.array dual_x: (m, 1) Dual variables.
        :param np.array constraint_array: (m, 1) Values of each of the constraints.
        :param np.array active_constraints: (m, 1) Bool values, True if active, False if inactive. If None, then all
            are assumed to be active.
        :return np.array: (n + p, 1) Gradient of the Lagrangian of the problem, p is the number of active constraints.
        """
        grad = grad_f
        constraint_grads = self.constraint.get_gradient(x)
        dual_2 = dual_x * 1.0
        dual_2[np.logical_not(constraint_array)] = 0.
        for i, gi in enumerate(constraint_grads):
            grad = grad + float(dual_x[i]) * gi
        if active_constraints is None:
            ca_active = constraint_array
        else:
            # Don't consider any of the constraint values if the are inactive (i.e., g(x)_i <= 0)
            ca_active = (constraint_array[active_constraints]).reshape(-1, 1)
        if len(ca_active) != 0:
            grad = np.row_stack((grad, ca_active))
        return grad

    def get_constraint_array(self, x):
        """ Returns the column vector of constraint values. """
        return np.array(self.constraint.get_g(x)).reshape((-1, 1))

    def get_constraint_gradient_array(self, x):
        """ Returns the column stack of the gradients of each constraint.

        :param np.array x: (n, 1) Primal variables.
        :return np.array: (n, m) Gradients of all the constraint functions.

        The returned array is equivalent with the transpose of the Jacobian of the constraint vector function.
        """
        all_constraint_grads = self.constraint.get_gradient(x)
        constraint_grads = 1. * all_constraint_grads[0]
        for i in range(1, len(all_constraint_grads)):
            constraint_grads = np.column_stack((constraint_grads, 1. * all_constraint_grads[i]))
        return constraint_grads

    def solve(self, x_0, dual_x_0):
        """ Returns the variables and dual variables that minimize the objective function s.t. the constraints.

        :param np.array x_0: (n, 1) Initial guess at primal variables.
        :param np.array dual_x_0: (m, 1) Initial guess at dual variables, m is the number of constraints specified.
        :return list: Solution to the optimization problem.
        """
        # Sanitize the inputs
        if type(x_0) is not np.ndarray or type(dual_x_0) is not np.ndarray:
            x_0 = np.array(x_0)
            dual_x_0 = np.array(dual_x_0)
        # Make sure that the arrays are column vectors
        x_0 = x_0.reshape(-1, 1)
        dual_x_0 = dual_x_0.reshape(-1, 1)

        print "Starting SQP minimization..."
        [x, dual_x, exit_info] = self.globalized_sqp(x_0, dual_x_0)
        conv_criteria = exit_info['val']

        print exit_info['msg']
        print "Exiting with ||grad[L]|| = {0:e}".format(conv_criteria)
        print "x = {0}".format(x.reshape(-1))
        print "dual_x = {0}".format(dual_x.reshape(-1))

        return [x, dual_x]

    def solve_return_conv(self, x_0, dual_x_0):
        """ Returns the variables and dual variables that minimize the objective function s.t. the constraints.

        :param np.array x_0: (n, 1) Initial guess at primal variables.
        :param np.array dual_x_0: (m, 1) Initial guess at dual variables, m is the number of constraints specified.
        :return list: Solution to the optimization problem, also returns the convergence criteria at exit.
        """
        # Sanitize the inputs
        if type(x_0) is not np.ndarray or type(dual_x_0) is not np.ndarray:
            x_0 = np.array(x_0)
            dual_x_0 = np.array(dual_x_0)
        # Make sure that the arrays are column vectors
        x_0 = x_0.reshape(-1, 1)
        dual_x_0 = dual_x_0.reshape(-1, 1)

        print "Starting SQP minimization..."
        [x, dual_x, exit_info] = self.globalized_sqp(x_0, dual_x_0)
        convergence_criteria = exit_info['val']

        print exit_info['msg']
        print "Exiting with ||grad[L]|| = {0:e}".format(convergence_criteria)
        print "x = {0}".format(x.reshape(-1))
        print "dual_x = {0}".format(dual_x.reshape(-1))

        return [x, dual_x, convergence_criteria]
