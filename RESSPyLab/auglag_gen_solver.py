"""@package aug_lag_gen_solver
Augmented Lagrangian solver.
"""
import numpy as np
from numdifftools import nd_algopy as nda
from barrier_function import BarrierFunction
from posdef_check import posdef_check

EPS_MACHINE = 10. * np.finfo(float).eps  # set to be slightly larger than machine epsilon


def min_trust_radius(x):
    """ For testing if trust region radius is too small. """
    return EPS_MACHINE * np.linalg.norm(x)


class AugLagGenSolver:
    """ Solves constrained optimization problems using an augmented Lagrangian formulation. """

    def __init__(self, test_data, error_function, subproblem_solver, dumper=None, barrier_function=None,
                 accept_approx_its=True, verbose_dump=False):
        """ Implements an augmented Lagrangian optimization procedure to find the parameters for material models.

        :param list test_data: Test data over which to optimize the parameters.
        :param function error_function: Provides the error from a single test.
        :param function subproblem_solver: Provides a solution to the trust-region subproblem.
        :param GenSolverDumper dumper: Periodically dumps the optimization status.
        :param BarrierFunction barrier_function: Specify the type of barrier function to use, if None then no barrier
            is used.
        :param bool accept_approx_its: Specify whether or not to also use the secondary convergence criteria.

        - The test data should be in DataFrame format with a column labeled "e_true" and "Sigma_true".
        - The material model is defined by the error_function definition. error_function = f(x, test_data), where x
            are the model parameters.
        - The subproblem_solver is a function returning a float, f(h, g, Delta), where h is a Hessian matrix,
            g is a gradient vector, and Delta is the trust-region radius.
        """
        self.array_clean_tests = test_data
        self.error_function = error_function
        self.total_iterations = 0
        self.maximum_total_iterations = int(1e6)
        self.maximum_ntr_iterations = int(1e6)
        self.maximum_auglag_iterations = 20  # needs to be sufficiently larger than the exponent of auglag_tolerance
        self.subproblem_solver = subproblem_solver
        self.auglag_tolerance = np.sqrt(EPS_MACHINE / 10.)  # around 1.e-8
        self.verbose_dump = verbose_dump

        # Barrier function initialization
        if barrier_function is None:
            self.use_barrier = False
        elif isinstance(barrier_function, BarrierFunction):
            self.use_barrier = True
            self.barrier = barrier_function
        else:
            raise RuntimeError("Improper barrier function specified.")

        self.accepting_approx_its = accept_approx_its

        if dumper is None:
            self.use_dumper = False
        else:
            self.use_dumper = True
            self.dumper = dumper

    def set_maximum_iterations(self, n_ntr, n_auglag, n_total):
        """ Sets the maximum iterations in the NTR and augmented Lagrangian steps. """
        self.maximum_ntr_iterations = n_ntr
        self.maximum_auglag_iterations = n_auglag
        self.maximum_total_iterations = n_total
        return

    def set_auglag_tol(self, tol):
        self.auglag_tolerance = tol
        return

    def error_ensemble_nda(self, x_sol):
        """ Returns the total relative error from all the tests considered.

        :param np.array x_sol: Voce-Chaboche model parameters.
        :return float: total relative error.

        - A barrier function is used to constrain the values greater than 0 if self.use_barrier == True.
        """
        error_ensemble = 0.
        for cleanedTest in self.array_clean_tests:
            error_ensemble = error_ensemble + self.error_function(x_sol, cleanedTest)

        # Barrier function definition
        if self.use_barrier:
            for x in x_sol:
                error_ensemble = error_ensemble + self.barrier.value(x)

        return error_ensemble

    def lagrangian(self, x, c, miu, constraint):
        """ Returns the Lagrangian.

        :param np.array x: Updated V-C model parameters.
        :param float c: penalty parameter.
        :param np.array miu: Lagrange multiplier values, len(miu) = len(g) in constraint.
        :param AugLagConstraint constraint: Contains the constraints specified in the model.
        :return float: Lagrangian value.
        """
        total_error = self.error_ensemble_nda(x)

        # Inequality constraints
        g = constraint.get_g(x) * 1.0
        lagrangian = total_error
        for j, miu_j in enumerate(miu):
            l_coef = max(0., miu_j + c * g[j])
            lagrangian += 1. / (2. * c) * (l_coef ** 2 - miu_j ** 2)

        return lagrangian

    def grad_lagrangian(self, x, grad_error_function, c, miu, constraint):
        """ Returns the gradient vector of the Lagrangian.

        :param np.array x: Updated V-C model parameters.
        :param function grad_error_function: Provides the gradient of the error from a single test.
        :param float c: penalty parameter.
        :param np.array miu: Lagrange multiplier values, len(miu) = len(g) in constraint.
        :param AugLagConstraint constraint: Contains the constraints specified in the model.
        :return np.array: (n, 1) Gradient column vector, n = len(x).
        """
        d_lagrangian = np.reshape(grad_error_function(x) * 1.0, (len(x), 1)) * 1.0

        # Inequality constraints
        g = constraint.get_g(x)
        dg = constraint.get_gradient(x)
        for j, miu_j in enumerate(miu):
            d_coef = max(0., miu_j + c * g[j])
            d_lagrangian += d_coef * np.reshape(dg[j], (len(x), 1)) * 1.0

        return d_lagrangian

    def hess_lagrangian(self, x, hess_error_function, c, miu, constraint):
        """ Returns the gradient vector of the Lagrangian.

        :param np.array x: Updated V-C model parameters.
        :param function hess_error_function: Provides the Hessian of the error from a single test.
        :param float c: penalty parameter.
        :param np.array miu: Lagrange multiplier values, len(miu) = len(g) in constraint.
        :param AugLagConstraint constraint: Contains the constraints specified in the model.
        :return np.array: (n, n) Hessian matrix, n = len(x)
        """
        h_lagrangian = hess_error_function(x) * 1.0

        # Inequality constraints
        g = constraint.get_g(x)
        dg = constraint.get_gradient(x)
        hg = constraint.get_hessian(x)
        for j, miu_j in enumerate(miu):
            h_coef1 = float(max(0., miu_j + c * g[j]))
            if h_coef1 > 0:
                h_coef2 = float(c * g[j])
                h_lagrangian += h_coef1 * hg[j] + h_coef2 * np.outer(dg[j], dg[j])

        return h_lagrangian

    def ntr_j_solver_lag(self, x, lagrangian_args, constraint, grad_error_fun, hess_error_fun):
        """ Newton trust region solver.

        :param np.array x: Initial, feasible point.
        :param dict lagrangian_args: Arguments required for the Lagrangian functions and derivatives.
        :param Constraint constraint: Defines the constraints that will be applied to the optimization model.
        :param function grad_error_fun: Function that returns the gradient of the error function.
        :param function hess_error_fun: Function that returns the Hessian of the error function.
        :return list: [x, Delta] solution point, and trust region radius at exit.
        """
        # Tolerance for ill-conditioned cases. Accepts the solution as an approximation if the trust-region radius
        # and the scaled step size are too small (TOL_Approx).
        tol_approx = 1.0e-3
        max_iterations = self.maximum_ntr_iterations

        # Quality close to 1 or greater = very good; close to 0 or smaller = poor; in-between = good
        poor_quality = 0.01
        good_quality = 0.9
        amazing_quality = 1000

        # Set-up the Lagrangian
        c = lagrangian_args['c']
        delta = lagrangian_args['Delta']
        miu = lagrangian_args['miu']
        tol = lagrangian_args['tol_k']
        grad_lagrangian = self.grad_lagrangian(x, grad_error_fun, c, miu, constraint) * 1.0
        hess_lagrangian = self.hess_lagrangian(x, hess_error_fun, c, miu, constraint) * 1.0

        # Calculate the optimal x parameters
        for nit in range(max_iterations):
            hess_old = hess_lagrangian  # this is only for the dumper
            # Update the constraint and calculate the value of the Lagrangian
            constraint.update_variables(x)
            lagrangian = self.lagrangian(x, c, miu, constraint) * 1.0

            # Solve trust-region sub-problem with steihaug-toint method and a scaled hessian
            diag_matrix = np.power(np.abs(np.diag(hess_lagrangian)), 0.5)
            diag_matrix[np.abs(diag_matrix) < 1e-10 * np.max(diag_matrix)] = 1
            preconditioner = np.diag(1 / diag_matrix)  # Jacobi preconditioner
            hess_lagrangian_s = np.dot(np.dot(preconditioner.transpose(), hess_lagrangian), preconditioner)
            grad_lagrangian_s = np.dot(preconditioner.transpose(), grad_lagrangian)

            # Ensure that x never goes below 0 if using a barrier function
            feasible_step = False
            while not feasible_step:
                x_trajectory = self.subproblem_solver(hess_lagrangian_s, grad_lagrangian_s, delta)  # d_k in Bierlaire

                # Bring back the scaled step
                x_trajectory = np.dot(preconditioner, x_trajectory)
                x_trial = x * 1.0 + x_trajectory.reshape(-1)

                # Restrict the trust radius if steps below 0
                # See Conn et al. (2000) and Byrd et al. (2003) for the inspiration for this barrier method
                if self.use_barrier is False:
                    feasible_step = True
                elif np.min(x_trial) < 0.:
                    delta = 0.5 * delta
                    if delta < min_trust_radius(x):
                        break
                else:
                    feasible_step = True

            # Test the trial step, see pg. 794 of Conn et al. (2000)
            small_value = EPS_MACHINE * np.max([np.abs(lagrangian), 1.0])
            model_diff = -float(np.dot(x_trajectory.transpose(), grad_lagrangian)
                                + 0.5 * np.dot(x_trajectory.transpose(), np.matmul(hess_lagrangian, x_trajectory)))
            constraint.update_variables(x_trial)
            lagrangian_trial = self.lagrangian(x_trial, c, miu, constraint) * 1.0
            lagrangian_diff = -float(lagrangian_trial - lagrangian)
            if np.abs(model_diff + small_value) < EPS_MACHINE and np.abs(lagrangian_diff + small_value) < EPS_MACHINE:
                model_quality = 1.
            else:
                model_quality = float((lagrangian_diff + small_value) / (model_diff + small_value))

            # Failure: poor quality or Lagrangian is increasing, restrict the trust region and don't update x
            if model_quality < poor_quality or lagrangian_diff < 0 or model_quality > amazing_quality:
                preconditioner_sqrt = np.diag(np.power(np.abs(np.diag(hess_lagrangian)), 0.5))
                delta = 0.5 * np.linalg.norm(np.dot(preconditioner_sqrt, x_trajectory))

            # Success: update x and check the trust region
            else:
                x = x_trial * 1.0
                grad_lagrangian = self.grad_lagrangian(x, grad_error_fun, c, miu, constraint)
                hess_lagrangian = self.hess_lagrangian(x, hess_error_fun, c, miu, constraint)

                if model_quality >= good_quality:
                    delta = 2. * delta

            # Update the barrier parameter
            if self.use_barrier:
                self.barrier.update_barrier_parameter()

            # Update the global iteration count
            self.total_iterations += 1

            # Periodically dump results
            convergence_criteria = np.linalg.norm(grad_lagrangian)
            if self.use_dumper:
                if self.verbose_dump:
                    dump_info = {'x': x, 'it_num': self.total_iterations, 'f_val': lagrangian,
                                 'norm_grad_lag': convergence_criteria, 'dk': x_trajectory, 'Delta': delta,
                                 'g': constraint.get_g(x), 'grad': grad_lagrangian,
                                 'hess_cond': np.linalg.cond(hess_old, p=2),
                                 'hess_s_cond': np.linalg.cond(hess_lagrangian_s, p=2),
                                 'hess_posdef': posdef_check(hess_lagrangian)}
                else:
                    dump_info = {'x': x, 'it_num': self.total_iterations, 'f_val': lagrangian,
                                 'norm_grad_lag': convergence_criteria}
                self.dumper.dump(dump_info)

            # Check convergence
            if convergence_criteria < tol:
                break
            elif delta < min_trust_radius(x):
                print " WARNING: exiting NTR-J because of too small search radius."
                break
            elif self.accepting_approx_its and np.linalg.norm(x_trajectory) < tol_approx and delta < tol_approx:
                print " WARNING: SECONDARY CONVERGENCE CRITERIA TRIGGERED. NORM OF GRADIENT NOT WITHIN TOLERANCE, " \
                      "THUS ONLY AN APPROXIMATE SOLUTION IS OBTAINED"
                break

        return [x, delta, nit]

    def augmented_lagrangian_opt(self, x, constraint):
        """ Augmented Lagrangian optimization procedure for inequality constraints.

        :param np.array x: Initial value of updated Voce-Chaboche model parameters.
        :param Constraint constraint: Constraints to the Augmented Lagrangian Optimization procedure.
        :return np.array: Optimized Voce-Chaboche model parameters.

        -The order of the parameters is: [E, sy0, qInf, b, dInf, a, C1, gamma1, ..., CN, gammaN] for N backstresses.
        -The strain column in the DataFrames should be labeled as "e_true", the stress should be labled as "Sigma_true".

        The Augmented Lagrangian procedure used in this function follows Bierlaire 2015 and Bertsekas 2016.
        """
        # Define constant values
        lagrangian_step_limit = self.maximum_auglag_iterations
        tol = self.auglag_tolerance
        # Tolerance for ill-conditioned cases. Accepts the solution as an approximation if the trust-region radius and
        # the scaled step size are too small (TOL_Approx)
        tol_approx = 1e-3
        approx_it_limit = 15  # only accept a limited number of ill-conditioned iterations.

        # Trust region radius
        trust_radius = 10.0  # \Delta in Bierlaire

        # Augmented lagrangian parameters
        penalty_parameter = 10.  # c in Bierlaire
        eta_hat_zero = 0.1258925
        tau = 10
        alpha = 0.1
        beta = 0.9
        tol_0 = 1. / penalty_parameter
        ntr_precision = 1. / penalty_parameter
        eta_al = eta_hat_zero / penalty_parameter ** alpha

        # Make sure the constraint is initialized
        constraint.update_variables(x)
        g = constraint.get_g(x)

        # Inequality Lagrange multipliers
        miu = np.zeros(len(g))

        # Arguments for the NTR procedure
        ntr_parameters = {'c': penalty_parameter, 'Delta': trust_radius, 'miu': miu, 'tol_k': ntr_precision}

        # Define the error function, and the gradient / Hessian for each set of test data with AlgoPy
        grad_error_fun = nda.Gradient(self.error_ensemble_nda)
        hess_error_fun = nda.Hessian(self.error_ensemble_nda)

        # Reset the total iterations to zero
        self.total_iterations = 0

        # Run the optimization
        num_of_approx_it = 0
        for i in range(int(lagrangian_step_limit)):
            print("##########      New Lagrangian Step      ###########")

            # Solve the trust region sub-problem
            x_init = x * 1.0
            ntr_parameters['Delta'] = trust_radius  # this is only here because of the approximate tolerance
            ntr_parameters['c'] = penalty_parameter
            ntr_parameters['miu'] = miu
            ntr_parameters['tol_k'] = ntr_precision
            [x, Delta2, nit] = self.ntr_j_solver_lag(x, ntr_parameters, constraint, grad_error_fun, hess_error_fun)

            # Update the constraints
            constraint.update_variables(x)
            g = constraint.get_g(x)
            g_max = np.maximum(np.zeros(np.shape(g)), g)
            ineqaulity_constraint_norm = np.linalg.norm(g_max)

            if ineqaulity_constraint_norm <= eta_al:
                # Update the Lagrange multipliers
                for j in range(len(miu)):
                    miu[j] = max(0, miu[j] + penalty_parameter * g[j])
                ntr_precision = ntr_precision / penalty_parameter
                eta_al = eta_al / penalty_parameter ** beta
            else:
                # Update the penalty parameter
                penalty_parameter = tau * penalty_parameter
                ntr_precision = tol_0 / penalty_parameter
                eta_al = eta_hat_zero / penalty_parameter ** alpha

            # Check convergence
            step_size = np.linalg.norm(x - x_init) * 1.0
            grad_lagrangian = self.grad_lagrangian(x, grad_error_fun, penalty_parameter, miu, constraint)

            # Check that gradient is nearly 0, and all the constraints are satisfied
            if np.linalg.norm(grad_lagrangian) < tol and ineqaulity_constraint_norm < tol:
                print "####################################################"
                print "### SUCCESSFUL AUGMENTED LAGRANGIAN OPTIMIZATION ###"
                print "####################################################"
                print "########## TERMINATING AUGMENTED LAGRANGIAN ########"
                print "####################################################"
                print ("x = ", x)
                break
            elif Delta2 < min_trust_radius(x):
                print "####################################################"
                print "###### EXITING BECAUSE OF TRUST REGION RADIUS ######"
                print "####################################################"
                print ("x = ", x)
                break
            elif self.total_iterations >= self.maximum_total_iterations:
                print "####################################################"
                print "######## EXITING BECAUSE OF TOTAL ITERATIONS #######"
                print "####################################################"
                print ("x = ", x)
                break
            elif self.accepting_approx_its and step_size < tol_approx and Delta2 < tol_approx:
                print " WARNING: SECONDARY CONVERGENCE CRITERIA TRIGGERED. NORM OF GRADIENT NOT WITHIN TOLERANCE."
                print " ONLY AN APPROXIMATE SOLUTION IS OBTAINED"
                num_of_approx_it = num_of_approx_it + 1
                trust_radius = 1.0
                if num_of_approx_it > approx_it_limit:
                    print "####################################"
                    print "# TERMINATING AUGMENTED LAGRANGIAN #"
                    print "####################################"
                    print ("x = ", x)
                    break

        return x
