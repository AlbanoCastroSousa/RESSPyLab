"""@package sqp_trustregion
Implements the SQP trust-region method.
"""
import numpy as np
import numpy.linalg as la
import warnings

from sqp_solver import SqpSolver
from collections import OrderedDict
from reduced_cg import reduced_cg


class SqpTrustregion(SqpSolver):

    def __init__(self, objective_function, constraint, dumper=None):
        """ Uses an SQP trust-region method to solve the minimization problem.

        :param EqFromIneqConstraint constraint: Equality constraints generated from inequality constraints using slack
            variables.

        See the SqpSolver mother class for definition of the other input parameters, and further details. In general
        the Byrd-Omojokun trust-region SQP method from [1] is used. The QP subproblem is solved using the reduced
        conjugate gradient with an imposed trust-region constraint and a guard against negative curvature. For futher
        details see the references below.

        References:
            [1] Nocedal and Wright (2006) "Numerical optimization"
            [2] Byrd et al. (1992) "An Interior Point Algorithm for Large-Scale Nonlinear Programming"
            [3] Conn et al. (2000) "Trust-region methods"
            [4] Bierlaire (2015) "Optimization: Principles and Algorithms"
        """
        SqpSolver.__init__(self, objective_function, constraint, dumper)
        self.too_small_trust_radius_exit = 3
        self.too_large_trust_radius_exit = 4
        return

    def calc_penalty_parameter(self, x, d_x, dual_x, grad_f, hess_l, previous_penalty):
        """ Returns the penalty parameter.

        :param np.array x: (n+m, 1) Primal variables, including slack variables.
        :param np.array d_x: (n+m, 1) Step in primal variables, including slack variables.
        :param np.array grad_f: (n, 1) Gradient of the objective function.
        :param np.array hess_l: (n, n) Hessian of the Lagrangian with respect to x.
        :param float previous_penalty: Previous value of the penalty parameter.
        :return float: New value of the penalty parameter.

        See [1] pg. 542 for details.
        """
        rho = 0.3  # From [2] pg. 891
        constraint_array = self.get_constraint_array(x + d_x)
        curvature = np.dot(d_x.transpose(), np.matmul(hess_l, d_x))
        if curvature > 0:
            sigma = 1
        else:
            sigma = 0
        # Take the max to avoid divide by zero error and ensure that the denominator doesn't "blow up"
        constr_violation = max(1e-8, la.norm(constraint_array, 2))
        penalty_trial = (np.dot(grad_f.transpose(), d_x) + (sigma / 2.) * curvature) / ((1. - rho) * constr_violation)

        if previous_penalty > penalty_trial:
            penalty_parameter = previous_penalty
        else:
            penalty_parameter = 0.1 + penalty_trial
        return float(penalty_parameter)

    def merit_fun(self, x, c):
        """ Nonsmooth, exact, L-2 merit function.

        :param np.array x: Primal variables.
        :param float c: Penalty parameter.
        """
        # Don't consider active constraints here since we are using an equality constraints through slack variables
        ca = self.get_constraint_array(x)
        x_p, _ = self.constraint.get_primals_slacks(x)
        return float(self.objective_fun.value(x_p) + c * la.norm(ca, 2))

    def calc_model_quality(self, x, d_x, grad_f, hess_l, penalty_parameter):
        """ Returns an indication of how well the quadratic model represents the objective function.

        :param np.array x: (n+m, 1) Primal variables, including slack variables.
        :param np.array d_x: (n+m, 1) Step in primal variables.
        :param np.array grad_f: (n, 1) Gradient of the objective funciton.
        :param np.array hess_l: (n, n) Hessian of the Lagrangian with respect to x.
        :param float penalty_parameter: Penalty on the constraint violation.
        :return float: Model quality, near 0 is bad quality and around 1 is very good quality.
        """
        constraint_array = self.get_constraint_array(x)
        jacobian_g = self.get_constraint_gradient_array(x).transpose()

        # Used for the actual reduction in the merit function
        merit_old = self.merit_fun(x, penalty_parameter)
        merit_trial = self.merit_fun(x + d_x, penalty_parameter)

        # Calculate the predicted reduction from the model
        model_old = la.norm(constraint_array, 2)
        model_trial = float(np.dot(grad_f.transpose(), d_x) + 0.5 * np.dot(d_x.transpose(), np.matmul(hess_l, d_x))
                            + penalty_parameter * la.norm(constraint_array + np.matmul(jacobian_g, d_x), 2))

        # See [3] pg. 794 for robust calculation of the model quality
        e = 10. * np.finfo(float).eps
        x_p, _ = self.constraint.get_primals_slacks(x)
        f_val = self.objective_fun.value(x_p)
        small_value = e * max(1., abs(f_val))
        model_diff = -model_old + model_trial - small_value
        merit_diff = -merit_old + merit_trial - small_value
        if abs(merit_diff) <= e and abs(model_diff) <= e:
            model_quality = 1.
        else:
            model_quality = merit_diff / model_diff

        return model_quality

    def globalized_sqp(self, x_0, dual_x_0):
        """ Globally convergence SQP trust-region algorithm.

        :param np.array x_0: (n, 1) Initial estimate of primal variables.
        :param np.array dual_x_0: (n, 1) Initial estimate of Lagrange multipliers.
        :return list: [x, dual_x] Vectors of shape=(n, 1), the optimized primal and dual variables.

        - A tolerance of 1e-5 is used for ||grad[L]|| (although this is high for typical implementations), the reason
            is that due to round-off errors in the Lagrange multipliers a lower tolerance is not typically reached. For
            the test problems investigated, a relative difference < 1.e6 is still found using this implementation. In
            this case, often the maximum trust-region radius is reached after many successful iterations that do not
            tend to change the solution point or gradient.
        """
        # Initialization
        maximum_iterations = self.maximum_iterations
        minimum_trust_radius = 1.e-8
        maximum_trust_radius = 1.e16
        tol = 1.e-5
        slack_active_tol = 1.e-2

        # Trust region parameters
        poor_quality = 0.01
        good_quality = 0.9
        radius_increase = 2.
        radius_decrease = 2.
        trust_region_radius = 10.

        x = self.constraint.calc_initial_x_slacks(x_0)
        dual_x = dual_x_0
        # Don't let the penalty be too big at the start since it never decreases
        penalty_parameter = max(1.0, la.norm(dual_x, np.inf) + 0.1)  # based on [4] pg. 480
        successful_step = True
        while self.total_iterations < maximum_iterations:
            # Update values upon a successful step
            if successful_step:
                x_primals, x_slacks = self.constraint.get_primals_slacks(x)
                function_value = self.objective_fun.value(x_primals)
                constraint_array = self.get_constraint_array(x)
                grad_f = self.constraint.append_0_slacks(self.objective_fun.grad(x_primals))
                hess_f = self.constraint.append_0_slacks_matrix(self.objective_fun.hess(x_primals))
                constraint_jacobian = self.get_constraint_gradient_array(x).transpose()
                # When the slack variables go to zero then the constraint is active
                active_index = np.abs(x_slacks) < slack_active_tol
                grad_l = self.grad_lagrangian(x, grad_f, dual_x, constraint_array, active_index)
                convergence_criteria = la.norm(grad_l)

                # Tell the user if the constraint Jacobian is rank deficient (linearly dependent constraints)
                a_rank = la.matrix_rank(constraint_jacobian)
                a_row, a_col = np.shape(constraint_jacobian)
                if a_row < a_rank:
                    warnings.warn('Constraint Jacobian is rank deficient.', RuntimeWarning)

            # Dump the progress when appropriate
            if self.use_dumper:
                dump_info = OrderedDict([('it_num', self.total_iterations), ('f_val', function_value),
                                         ('norm_grad_lag', convergence_criteria), ('x', x_primals),
                                         ('dual_x', dual_x), ('trust_radius', trust_region_radius),
                                         ('penalty', penalty_parameter),
                                         ('g(x)', constraint_array)])
                self.dumper.dump(dump_info)

            # Check convergence and return results
            if convergence_criteria < tol and la.norm(constraint_array, np.inf) < tol:
                break

            # Calculate a trial step
            hess_l = self.hess_xx_lagrangian(x, hess_f, dual_x)
            [x_step, dual_x_step] = reduced_cg(hess_l, grad_f, constraint_jacobian, -constraint_array,
                                               trust_region_radius)
            # From the reduced_cg we get the negative of the Lagrange multipliers, so modify by -1.
            dual_x_step = -1.0 * dual_x_step

            # Test the trial step
            penalty_parameter = self.calc_penalty_parameter(x, x_step, dual_x_step, grad_f, hess_l, penalty_parameter)
            model_quality = self.calc_model_quality(x, x_step, grad_f, hess_f, penalty_parameter)
            if model_quality < poor_quality:
                # Model is poor, reject step and reduce the radius
                trust_region_radius = trust_region_radius / radius_decrease
                successful_step = False
                if trust_region_radius < minimum_trust_radius:
                    break
            else:
                # Model is good, accept the step
                x = x + x_step
                dual_x = dual_x_step
                successful_step = True
                if model_quality >= good_quality:
                    # Model is very good, increase the radius
                    trust_region_radius = trust_region_radius * radius_increase
                    if trust_region_radius > maximum_trust_radius:
                        break

            # Go to next iteration
            self.total_iterations += 1
        # END WHILE

        # Exited non-optimally
        if convergence_criteria < tol and la.norm(constraint_array, np.inf) < tol:
            exit_info = {'tag': self.convergence_reached_tag, 'val': convergence_criteria,
                         'msg': "SQP trust-region converged in {0} iterations.".format(self.total_iterations)}
        elif self.total_iterations >= maximum_iterations:
            exit_info = {'tag': self.maximum_iterations_reached_tag, 'val': convergence_criteria,
                         'msg': "\nMaximum iterations reached in SQP"}
        elif trust_region_radius < minimum_trust_radius:
            exit_info = {'tag': self.too_small_trust_radius_exit, 'val': convergence_criteria,
                         'msg': "\nMinimum trust-region radius reached in SQP"}
        elif trust_region_radius > maximum_trust_radius:
            exit_info = {'tag': self.too_large_trust_radius_exit, 'val': convergence_criteria,
                         'msg': "\nMaximum trust-region radius reached in SQP"}
        else:
            exit_info = {'tag': self.unknown_exit, 'val': convergence_criteria,
                         'msg': "Unknown exit condition reached"}

        x, _ = self.constraint.get_primals_slacks(x)
        return [x, dual_x, exit_info]

    def solve(self, x_0, dual_x_0):
        # Need to let the constraint know how many primal variables there are
        self.constraint.set_n_slacks_primals(x_0)
        return SqpSolver.solve(self, x_0, dual_x_0)
