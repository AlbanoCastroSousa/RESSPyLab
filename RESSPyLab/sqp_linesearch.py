"""@package sqp_linsearch
Implements the SQP linesearch method.
"""
from sqp_solver import SqpSolver
import numpy as np
import numpy.linalg as la
from quadprog import solve_qp
from modified_cholesky import modified_cholesky
from collections import OrderedDict


class SqpLinesearch(SqpSolver):

    def __init__(self, objective_function, constraint, dumper=None):
        """ SQP minimization that employs a line-search and the l-1 non-smooth merit function for global convergence.

        :param MatModelErrorNda objective_function: Defines the objective function, and it's gradient and Hessian.
        :param AugLagConstraint constraint: Defines inequality constraints applied to the problem.
        """
        SqpSolver.__init__(self, objective_function, constraint, dumper)
        self.line_search_failure = 4

    def set_active_constraints(self, lagrange_multipliers):
        """ Active constraints are those that have non-zero Lagrange multipliers. """
        self.active_constraints_set = True
        self.active_constraints_index = lagrange_multipliers != 0.
        return

    def get_active_constraints(self):
        """ Returns indices for the active constraints.

        :return np.array : (m, 1) Array of bools, True for the active constraints and False for the non-active.

        Use this function instead of the class member to ensure that the indicies have been set.
        """
        if self.active_constraints_set:
            return self.active_constraints_index
        else:
            raise Exception('Active constraints not set yet!')

    def get_active_constraint_array(self, x, active_constraints=None):
        c = self.get_constraint_array(x)
        return c[active_constraints]

    def merit_fun(self, x, c):
        """ Nonsmooth, exact, L-1 merit function.

        :param np.array x: Primal variables.
        :param float c: Penalty parameter.
        """
        ca = self.get_constraint_array(x)
        ca_active = ca[self.get_active_constraints()]
        return float(self.objective_fun.value(x) + c * la.norm(ca_active, 1))

    def quadprog(self, x, hessian, gradient, constraint_array):
        """ Returns the primal and dual solutions to the QP subproblem.

        :param np.array x: (n, 1) Primal variables.
        :param np.array hessian: (n, n) Hessian of the Lagrangian w.r.t. xx, assumed to be positive-definite.
        :param np.array gradient: (n, 1) Gradient of the objective function w.r.t x.
        :param np.array constraint_array: (m, 1) Values of each of the constraints.
        :return list: [(n, 1), (m, 1)] Primal and dual solution variables.


        Uses the quadprog package to solve the quadratic programming (QP) subproblem. See this package for more details.
        The form of the problem assumed in the package is:

        Minimize_x   1/2 x^T G x - a^T x
        Subject to   C.T x >= b

        The problem we want to solve is:

        Minimize_x   1/2 d^T H d + grad[f(x)]^T d
        Subject to   grad[h(x)]^T d + h(x) <= 0

        So we set:
            G = hessian (H)
            a = -gradient (-grad[f(x)])
            C = -constraint_grads (-grad[h(x)])
            b = constraint_array (h(x))
        The constraints are multiplied by -1. to obtain the form C.T <= b assumed in the SQP algorithm. Therefore,
        no factor is applied to constraint_array since the result is -1 * -1 * constraint_array to move it to the rhs.

        Note the "solve_qp" function states that the returned value "lagrangian" is the "vector with the Lagrangian at
        the solution", however more specifically this is the vector of Lagrange multipliers (dual variables). For the
        actual definition see the solve.QP.c file (https://github.com/rmcgibbo/quadprog/tree/master/quadprog, 29/08/18).
        """
        b = constraint_array.reshape(-1)
        if len(b) == 0:
            qp_solution = solve_qp(hessian, -1. * gradient.reshape(-1))
        else:
            constraint_grads = -1 * self.get_constraint_gradient_array(x)
            qp_solution = solve_qp(hessian, -1. * gradient.reshape(-1), constraint_grads, b)

        d_x = qp_solution[0]
        if len(b) > 0:
            d_lambda = qp_solution[4]
        else:
            d_lambda = np.array([])

        return [d_x.reshape(len(d_x), 1), d_lambda.reshape(len(d_lambda), 1)]

    def globalized_sqp(self, x_0, dual_x_0):
        """ Uses a globalized SQP method with a line-search to solve the minimization problem.

        :param np.array x_0: (n, 1) Initial guess at primal variables.
        :param np.array dual_x_0: (m, 1) Initial guess at dual variables, m is the number of constraints specified.
        :return list: Primal and dual solutions, and exit information.

        Follows Algorithm 20.2 from Bierlaire (2015) "Optimization: Principles and Algorithms", pg. 480.

        Raises a RuntimeError if the line-search algorithm does not converge.
        """
        # Initialization
        maximum_iterations = self.maximum_iterations
        tol = self.precision

        x = x_0
        dual_x = dual_x_0
        c_bar = 0.1  # basic penalty parameter value
        if len(dual_x) == 0:
            penalty_parameter = 0.
        else:
            penalty_parameter = la.norm(dual_x, ord=np.inf) + c_bar
        self.set_active_constraints(dual_x)
        constraint_array = self.get_constraint_array(x)
        grad_f = self.objective_fun.grad(x)
        hess_f = self.objective_fun.hess(x)
        convergence_criteria = la.norm(self.grad_lagrangian(x, grad_f, dual_x, constraint_array,
                                                            self.get_active_constraints()))

        # Calculate the primal and dual solutions
        while convergence_criteria > tol and self.total_iterations < maximum_iterations:
            # Set the Hessian and get a positive-definite approximation
            hess_lagrangian = self.hess_xx_lagrangian(x, hess_f, dual_x)
            [hess_posdef, id_factor] = modified_cholesky(hess_lagrangian)

            # Solve the quadratic programming sub-problem to get the step direction
            [x_step, dual_x_step] = self.quadprog(x, hess_posdef, grad_f, constraint_array)
            self.set_active_constraints(dual_x_step)

            # Update the penalty parameter
            if len(dual_x_0) == 0:
                c_upper_bound = 0.
                penalty_parameter = 0.
            else:
                c_upper_bound = la.norm(dual_x_step, np.inf)
            if penalty_parameter >= 1.1 * c_upper_bound:
                penalty_parameter = 0.5 * (penalty_parameter + c_upper_bound)
            # If c_upper_bound <= penalty_parameter < 1.1 * c_upper_bound -> don't change penalty_parameter
            elif penalty_parameter < c_upper_bound:
                penalty_parameter = np.max([1.5 * penalty_parameter, c_upper_bound])

            # Calculate the step length using a line-search
            active_constraints = constraint_array[self.active_constraints_index]
            merit_descent = float(np.dot(grad_f.transpose(), x_step)
                                  - penalty_parameter * la.norm(active_constraints, 1))
            [step_trajectory, step_size, ls_conv] = self.basic_linesearch(x, x_step, penalty_parameter, merit_descent)

            # Exit the solver if the line-search does not converge
            if not ls_conv:
                break

            # Update parameters for the next step
            x = x + step_trajectory
            dual_x = dual_x_step
            grad_f = self.objective_fun.grad(x)
            hess_f = self.objective_fun.hess(x)
            constraint_array = self.get_constraint_array(x)
            self.total_iterations += 1
            convergence_criteria = float(la.norm(self.grad_lagrangian(x, grad_f, dual_x, constraint_array,
                                                                      self.get_active_constraints())))

            # Dump the progress when appropriate
            if self.use_dumper:
                dump_info = OrderedDict([('it_num', self.total_iterations),
                                         ('step_factor', step_size),
                                         ('f_val', self.objective_fun.value(x)),
                                         ('norm_grad_lag', convergence_criteria),
                                         ('x', x)])
                self.dumper.dump(dump_info)

        # Let the solver know how it exited
        if convergence_criteria <= tol:
            exit_info = {'tag': self.convergence_reached_tag, 'val': convergence_criteria,
                         'msg': "SQP line-search converged in {0} iterations.".format(self.total_iterations)}
        elif self.total_iterations >= maximum_iterations:
            exit_info = {'tag': self.maximum_iterations_reached_tag, 'val': convergence_criteria,
                         'msg': "\nMaximum iterations reached in SQP."}
        elif not ls_conv:
            exit_info = {'tag': self.line_search_failure, 'val': convergence_criteria,
                         'its': self.total_iterations,
                         'msg': "\nLine search did not converge in 50 iterations."}
        else:
            exit_info = {'tag': self.unknown_exit, 'val': convergence_criteria,
                         'msg': "Unknown exit condition reached."}

        return [x, dual_x, exit_info]

    def basic_linesearch(self, x, d_x, c, merit_descent):
        """ Backtracking line-search using a full-Newton descent direction.

        :param np.array x: Primal variables.
        :param np.array d_x: Step in primal variables.
        :param float c: Penalty parameter.
        :param float merit_descent: Directional derivative of the merit function.
        :return float: The step length factor.

        Only the first Wolfe conditions are imposed because a back-tracking line-search strategy is used.
        See Nocedal and Wright (2006) Ch. 3 for more details.
        """
        max_line_its = 50  # this will give a minimum step length factor of around 8*10^-16
        sufficient_decrease_factor = 0.3
        step_length_decrease = 2.  # halve the step length at each failed iteration

        step_length = 1.
        merit_init = self.merit_fun(x, c)
        merit_trial = self.merit_fun(x + d_x, c)
        merit_check = merit_init + step_length * sufficient_decrease_factor * merit_descent
        line_search_iterations = 0
        line_search_converged = True
        while merit_trial > merit_check and line_search_iterations < max_line_its:
            step_length = step_length / step_length_decrease
            merit_trial = self.merit_fun(x + step_length * d_x, c)
            merit_check = merit_init + step_length * sufficient_decrease_factor * merit_descent
            line_search_iterations += 1

        # Raise an exception if the line-search did not converge
        if line_search_iterations >= max_line_its:
            line_search_converged = False

        return [d_x * step_length, step_length, line_search_converged]

    def corrected_linesearch(self, x, d_x, c, merit_descent):
        """ Similar to the basic line-search but with a 2nd order correction to alleviate the Maratos effect.

        Based on Algorithm 15.2 from Nocedal and Wright (2006), pg. 443-444.
        """
        max_line_its = 50  # this will give a minimum step length factor of around 8*10^-16
        sufficient_decrease_factor = 0.3
        step_length_decrease = 2.  # halve the step length at each failed iteration

        step_length = 1.
        merit_init = self.merit_fun(x, c)
        line_search_iterations = 0
        new_point = False
        used_correction = False
        correction = 0.
        line_search_converged = True
        while new_point is False and line_search_iterations < max_line_its:
            # Start with regular line search
            merit_trial = self.merit_fun(x + step_length * d_x, c)
            merit_check = merit_init + step_length * sufficient_decrease_factor * merit_descent
            if merit_trial <= merit_check:
                new_point = True
            elif step_length == 1.:
                # Apply the 2nd order correction only on the first iteration
                correction = self.calc_2nd_correction(x, d_x)
                merit_trial = self.merit_fun(x + d_x + correction, c)
                merit_check = merit_init + sufficient_decrease_factor * merit_descent  # step_length == 1 here
                if merit_trial <= merit_check:
                    used_correction = True
                    new_point = True
                else:
                    step_length = step_length / step_length_decrease
            else:
                # If the 2nd order correction doesn't give a sufficient decrease, backtrack on the original direction
                step_length = step_length / step_length_decrease

            line_search_iterations += 1

        if used_correction:
            dx_total = d_x + correction
            print "Used 2nd order correction in line-search."  # todo: remove this at some point
        else:
            dx_total = d_x * step_length

        # Raise an exception if the line-search did not converge
        if line_search_iterations >= max_line_its:
            line_search_converged = False

        return [dx_total, step_length, line_search_converged]

    def calc_2nd_correction(self, x, d_x):
        """ Calculates the 2nd order correction step.

        :param np.array x: (n, 1) Primal variables.
        :param np.array d_x: (n, 1) Step direction.
        :return np.array: (n, 1) 2nd order correction step.
        """
        # todo: not sure if this works with active constraints in the current formulation -> maybe doesn't do anything
        ca = self.get_constraint_array(x + d_x)
        active_index = self.get_active_constraints()
        ca_active = ca[active_index]
        if len(ca_active) == 0:
            d_second_order = 0.
        else:
            c_jacobian = self.get_constraint_gradient_array(x)
            c_jacobian = c_jacobian[:, active_index.reshape(-1)]
            if len(ca_active) == 1:
                # Only have one active constraint, need to adjust the matrix algebra since we get scalars
                c_jacobian = c_jacobian.reshape(1, -1)
                a = -1. * np.matmul(c_jacobian.transpose(), la.inv(np.matmul(c_jacobian, c_jacobian.transpose())))
                d_second_order = a * float(ca_active)
            else:
                c_jacobian = c_jacobian.transpose()
                a = -1. * np.matmul(c_jacobian.transpose(), la.inv(np.matmul(c_jacobian, c_jacobian.transpose())))
                d_second_order = np.matmul(a, ca)
        return d_second_order
