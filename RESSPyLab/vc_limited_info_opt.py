"""@package vco_limited_info_opt
Functions for limited information optimization using the original Voce-Chaboche model.
"""
from __future__ import print_function
import scipy.optimize as opt

from .RESSPyLab import errorTest_scl
from .mat_model_error_nda import MatModelErrorNda
from .scipy_dumper import ScipyBasicDumper
from .vc_li_opt_constraints import *
from .auglag_factory import auglag_factory, constrained_auglag_opt
from .scipy_constr_opt_factory import GWrapper
from .data_readers import load_and_filter_data_set, load_data_set
from .sqp_factory import sqp_factory


def vc_tensile_opt_scipy(x_0, file_list, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup,
                         rho_gamma_b_inf, rho_gamma_b_sup, rho_gamma_12_inf, rho_gamma_12_sup,
                         x_log_file='', fun_log_file='', filter_data=True,
                         max_its=600, tol=1.e-8, make_x0_feasible=True):
    """ Return parameters based on a single tensile test for the original VC model using the trust-constr method.

    :param np.array x_0: Initial primal variables.
    :param list file_list: [str] Path to the tensile test to use in the optimization.
    :param float rho_iso_inf: Lower bound on ratio of isotropic to total hardening at saturation.
    :param float rho_iso_sup: Upper bound on ratio of isotropic to total hardening at saturation.
    :param float rho_yield_inf: Lower bound on ratio of initial yield stress to total stress at saturation.
    :param float rho_yield_sup: Upper bound on ratio of initial yield stress to total stress at saturation.
    :param float rho_gamma_b_inf: Lower bound on ratio the rate of kinematic to isotropic hardening.
    :param float rho_gamma_b_sup: Upper bound on ratio the rate of kinematic to isotropic hardening.
    :param float rho_gamma_12_inf: Lower bound on ratio of gamma_1 to gamma_2.
    :param float rho_gamma_12_sup: Upper bound on ratio of gamma_1 to gamma_2.
    :param str x_log_file: Path to file to write the primal variable history.
    :param str fun_log_file: Path to file to write the objective function history.
    :param bool filter_data: If True, then filter data, else do not filter the data.
    :param int max_its: Maximum iterations allowed in analysis.
    :param float tol: Exit tolerance on the norm of grad[L].
    :param bool make_x0_feasible: If true then makes the first point feasible.
    :return list:
        - (np.array): Final primal variables.
        - (ScipyBasicDumper) Dumper used in analysis.
    """
    # Ensure x_0 is an array and not a list
    if isinstance(x_0, list):
        x_0 = np.array(x_0)

    # Load the data
    if filter_data:
        filtered_data = load_and_filter_data_set(file_list)
    else:
        filtered_data = load_data_set(file_list)

    # Define the objective function
    objective_function = MatModelErrorNda(errorTest_scl, filtered_data, use_cols=False)
    fun = objective_function.value
    jac = objective_function.grad
    hess = objective_function.hess

    # Set up the constraints for the optimization
    # Minimum value for any x_i
    min_x_bound = 0.01
    x_lb = [min_x_bound for i in range(len(x_0))]
    x_ub = [np.inf for i in range(len(x_0))]
    bounds = opt.Bounds(x_lb, x_ub, keep_feasible=True)
    # Bounds on inequality constraints
    constr_lb = -np.inf
    constr_ub = 0.

    # The constraint functions in scipy format
    g_constants = {'rho_yield_inf': rho_yield_inf, 'rho_yield_sup': rho_yield_sup,
                   'rho_iso_inf': rho_iso_inf, 'rho_iso_sup': rho_iso_sup,
                   'rho_gamma_inf': rho_gamma_b_inf, 'rho_gamma_sup': rho_gamma_b_sup,
                   'rho_gamma_12_inf': rho_gamma_12_inf, 'rho_gamma_12_sup': rho_gamma_12_sup}
    # g_3 - Constraint on ratio of saturation stress to initial yield stress
    g3_low = GWrapper(g3_vco_lower, g3_vco_lower_gradient, g3_vco_lower_hessian, g_constants)
    g3_high = GWrapper(g3_vco_upper, g3_vco_upper_gradient, g3_vco_upper_hessian, g_constants)
    g3_inf_constr = opt.NonlinearConstraint(g3_low.f, constr_lb, constr_ub, jac=g3_low.gf, hess=g3_low.hf)
    g3_sup_constr = opt.NonlinearConstraint(g3_high.f, constr_lb, constr_ub, jac=g3_high.gf, hess=g3_high.hf)
    # g_4 - Constraint on ratio of isotropic to kinematic hardening
    g4_low = GWrapper(g4_vco_lower, g4_vco_lower_gradient, g4_vco_lower_hessian, g_constants)
    g4_high = GWrapper(g4_vco_upper, g4_vco_upper_gradient, g4_vco_upper_hessian, g_constants)
    g4_inf_constr = opt.NonlinearConstraint(g4_low.f, constr_lb, constr_ub, jac=g4_low.gf, hess=g4_low.hf)
    g4_sup_constr = opt.NonlinearConstraint(g4_high.f, constr_lb, constr_ub, jac=g4_high.gf, hess=g4_high.hf)
    # g_5 - Constraint on rate of isotropic b to kinematic hardening gamma_1
    g5_low = GWrapper(g5_vco_lower, g5_vco_lower_gradient, g5_vco_lower_hessian, g_constants)
    g5_high = GWrapper(g5_vco_upper, g5_vco_upper_gradient, g5_vco_upper_hessian, g_constants)
    g5_inf_constr = opt.NonlinearConstraint(g5_low.f, constr_lb, constr_ub, jac=g5_low.gf, hess=g5_low.hf)
    g5_sup_constr = opt.NonlinearConstraint(g5_high.f, constr_lb, constr_ub, jac=g5_high.gf, hess=g5_high.hf)
    # g_6 - Constraint on ratio of backstress rates
    g_6_low = GWrapper(g6_vco_lower, g6_vco_lower_gradient, g6_vco_lower_hessian, g_constants)
    g_6_high = GWrapper(g6_vco_upper, g6_vco_upper_gradient, g6_vco_upper_hessian, g_constants)
    g6_inf_constr = opt.NonlinearConstraint(g_6_low.f, constr_lb, constr_ub, jac=g_6_low.gf, hess=g_6_low.hf)
    g6_sup_constr = opt.NonlinearConstraint(g_6_high.f, constr_lb, constr_ub, jac=g_6_high.gf, hess=g_6_high.hf)

    # Make the tuple of constraints
    n_backstresses = int((len(x_0) - 4) // 2)
    if n_backstresses == 2:
        constraints = (g3_inf_constr, g3_sup_constr, g4_inf_constr, g4_sup_constr, g5_inf_constr, g5_sup_constr,
                       g6_inf_constr, g6_sup_constr)
    elif n_backstresses == 1:
        constraints = (g3_inf_constr, g3_sup_constr, g4_inf_constr, g4_sup_constr, g5_inf_constr, g5_sup_constr)

    # Create a new dumper if None is provided
    dumper = ScipyBasicDumper(x_log_file, fun_log_file)

    # Make the point feasible
    if make_x0_feasible:
        x_0 = make_feasible(x_0, g_constants)

    # Run the minimization
    scipy_sol = opt.minimize(fun, np.array(x_0), method='trust-constr', jac=jac, hess=hess, bounds=bounds,
                             constraints=constraints, callback=dumper.dump,
                             options={'maxiter': max_its, 'verbose': 2, 'gtol': tol})
    # Check if the constraints were satisfied
    vc_constraint_check(scipy_sol.x, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup, rho_gamma_b_inf,
                        rho_gamma_b_sup)
    return [scipy_sol.x, dumper]


def vc_tensile_opt_auglag(x_0, file_list, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup,
                          rho_gamma_b_inf, rho_gamma_b_sup, rho_gamma_12_inf, rho_gamma_12_sup,
                          x_log_file='', fun_log_file='', filter_data=True,
                          max_its=600, tol=1.e-8, make_x0_feasible=True):
    """ Return parameters based on a single tensile test for the original VC model using the aug. Lagrangian method.

    :param np.array x_0: Initial primal variables.
    :param list file_list: [str] Path to the tensile test to use in the optimization.
    :param float rho_iso_inf: Lower bound on ratio of isotropic to total hardening at saturation.
    :param float rho_iso_sup: Upper bound on ratio of isotropic to total hardening at saturation.
    :param float rho_yield_inf: Lower bound on ratio of initial yield stress to total stress at saturation.
    :param float rho_yield_sup: Upper bound on ratio of initial yield stress to total stress at saturation.
    :param float rho_gamma_b_inf: Lower bound on ratio the rate of kinematic to isotropic hardening.
    :param float rho_gamma_b_sup: Upper bound on ratio the rate of kinematic to isotropic hardening.
    :param float rho_gamma_12_inf: Lower bound on ratio of gamma_1 to gamma_2.
    :param float rho_gamma_12_sup: Upper bound on ratio of gamma_1 to gamma_2.
    :param str x_log_file: Path to file to write the primal variable history.
    :param str fun_log_file: Path to file to write the objective function history.
    :param bool filter_data: If True, then filter data, else do not filter the data.
    :param int max_its: Maximum iterations allowed in a single trust-region sub-problem step.
    :param float tol: Exit tolerance on the norm of grad[L].
    :param bool make_x0_feasible: If true then makes the first point feasible.
    :return list:
        - (np.array): Final primal variables.
        - None

    Notes:
        - The use of vco_limited_info_opt_scipy() is recommended instead of this function.
    """
    # Load the data
    if filter_data:
        filtered_data = load_and_filter_data_set(file_list)
    else:
        filtered_data = load_data_set(file_list)

    # The constants for the constraint functions
    g_constants = {'rho_yield_inf': rho_yield_inf, 'rho_yield_sup': rho_yield_sup,
                   'rho_iso_inf': rho_iso_inf, 'rho_iso_sup': rho_iso_sup,
                   'rho_gamma_inf': rho_gamma_b_inf, 'rho_gamma_sup': rho_gamma_b_sup,
                   'rho_gamma_12_inf': rho_gamma_12_inf, 'rho_gamma_12_sup': rho_gamma_12_sup}
    # Set-up constraints
    n_backstresses = int((len(x_0) - 4) // 2)
    if n_backstresses == 2:
        constraint_dict = {'constants': g_constants, 'variables': {},
                           'functions': [g3_vco_lower, g3_vco_upper, g4_vco_lower, g4_vco_upper,
                                         g5_vco_lower, g5_vco_upper, g6_vco_lower, g6_vco_upper],
                           'gradients': [g3_vco_lower_gradient, g3_vco_upper_gradient, g4_vco_lower_gradient,
                                         g4_vco_upper_gradient, g5_vco_lower_gradient, g5_vco_upper_gradient,
                                         g6_vco_lower_gradient, g6_vco_upper_gradient],
                           'hessians': [g3_vco_lower_hessian, g3_vco_upper_hessian, g4_vco_lower_hessian,
                                        g4_vco_upper_hessian, g5_vco_lower_hessian, g5_vco_upper_hessian,
                                        g6_vco_lower_hessian, g6_vco_upper_hessian],
                           'updater': None}
    elif n_backstresses == 1:
        constraint_dict = {'constants': g_constants, 'variables': {},
                           'functions': [g3_vco_lower, g3_vco_upper, g4_vco_lower, g4_vco_upper,
                                         g5_vco_lower, g5_vco_upper],
                           'gradients': [g3_vco_lower_gradient, g3_vco_upper_gradient, g4_vco_lower_gradient,
                                         g4_vco_upper_gradient, g5_vco_lower_gradient, g5_vco_upper_gradient],
                           'hessians': [g3_vco_lower_hessian, g3_vco_upper_hessian, g4_vco_lower_hessian,
                                        g4_vco_upper_hessian, g5_vco_lower_hessian, g5_vco_upper_hessian],
                           'updater': None}
    else:
        raise ValueError('Only 1 or 2 backstresses are supported in the tensile calibration at the moment.')
    # Make the point feasible
    if make_x0_feasible:
        x_0 = make_feasible(x_0, g_constants)
    # Create the solver and run
    solver = auglag_factory(filtered_data, x_log_file, fun_log_file, 'original', 'steihaug', 'reciprocal', False)
    # max_its for each Lagrangian step, maximum of 20 Lagrangian steps, and allow 10 full steps (of max_its)
    solver.set_maximum_iterations(max_its, 20, max_its)
    solver.set_auglag_tol(tol)
    x_opt = constrained_auglag_opt(x_0, constraint_dict, solver)
    # Check if the constraints were satisfied
    vc_constraint_check(x_opt, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup, rho_gamma_b_inf,
                        rho_gamma_b_sup)
    return [DummyResults(x_opt), None]


def vc_tensile_opt_linesearch(x_0, file_list, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup,
                              rho_gamma_b_inf, rho_gamma_b_sup, rho_gamma_12_inf, rho_gamma_12_sup,
                              x_log_file='', fun_log_file='', filter_data=True,
                              max_its=600, tol=1.e-8, make_x0_feasible=True):
    """ Return parameters based on a single tensile test for the original VC model using the line search algorithm.

    :param np.array x_0: Initial primal variables.
    :param list file_list: [str] Path to the tensile test to use in the optimization.
    :param float rho_iso_inf: Lower bound on ratio of isotropic to total hardening at saturation.
    :param float rho_iso_sup: Upper bound on ratio of isotropic to total hardening at saturation.
    :param float rho_yield_inf: Lower bound on ratio of initial yield stress to total stress at saturation.
    :param float rho_yield_sup: Upper bound on ratio of initial yield stress to total stress at saturation.
    :param float rho_gamma_b_inf: Lower bound on ratio the rate of kinematic to isotropic hardening.
    :param float rho_gamma_b_sup: Upper bound on ratio the rate of kinematic to isotropic hardening.
    :param float rho_gamma_12_inf: Lower bound on ratio of gamma_1 to gamma_2.
    :param float rho_gamma_12_sup: Upper bound on ratio of gamma_1 to gamma_2.
    :param str x_log_file: Path to file to write the primal variable history.
    :param str fun_log_file: Path to file to write the objective function history.
    :param bool filter_data: If True, then filter data, else do not filter the data.
    :param int max_its: Maximum iterations allowed in analysis.
    :param float tol: Exit tolerance on the norm of grad[L].
    :param bool make_x0_feasible: If true then makes the first point feasible.
    :return list:
        - (np.array): Final primal variables.
        - None

    Notes:
        - The use of vco_limited_info_opt_scipy() is recommended instead of this function.
    """
    # Load the data
    if filter_data:
        filtered_data = load_and_filter_data_set(file_list)
    else:
        filtered_data = load_data_set(file_list)

    # The constants for the constraint functions
    g_constants = {'rho_yield_inf': rho_yield_inf, 'rho_yield_sup': rho_yield_sup,
                   'rho_iso_inf': rho_iso_inf, 'rho_iso_sup': rho_iso_sup,
                   'rho_gamma_inf': rho_gamma_b_inf, 'rho_gamma_sup': rho_gamma_b_sup,
                   'rho_gamma_12_inf': rho_gamma_12_inf, 'rho_gamma_12_sup': rho_gamma_12_sup}
    # Set-up constraints
    n_backstresses = int((len(x_0) - 4) // 2)
    if n_backstresses == 2:
        constraint_dict = {'constants': g_constants, 'variables': {},
                           'functions': [g3_vco_lower, g3_vco_upper, g4_vco_lower, g4_vco_upper,
                                         g5_vco_lower, g5_vco_upper, g6_vco_lower, g6_vco_upper],
                           'gradients': [g3_vco_lower_gradient, g3_vco_upper_gradient, g4_vco_lower_gradient,
                                         g4_vco_upper_gradient, g5_vco_lower_gradient, g5_vco_upper_gradient,
                                         g6_vco_lower_gradient, g6_vco_upper_gradient],
                           'hessians': [g3_vco_lower_hessian, g3_vco_upper_hessian, g4_vco_lower_hessian,
                                        g4_vco_upper_hessian, g5_vco_lower_hessian, g5_vco_upper_hessian,
                                        g6_vco_lower_hessian, g6_vco_upper_hessian],
                           'updater': None}
    elif n_backstresses == 1:
        constraint_dict = {'constants': g_constants, 'variables': {},
                           'functions': [g3_vco_lower, g3_vco_upper, g4_vco_lower, g4_vco_upper,
                                         g5_vco_lower, g5_vco_upper],
                           'gradients': [g3_vco_lower_gradient, g3_vco_upper_gradient, g4_vco_lower_gradient,
                                         g4_vco_upper_gradient, g5_vco_lower_gradient, g5_vco_upper_gradient],
                           'hessians': [g3_vco_lower_hessian, g3_vco_upper_hessian, g4_vco_lower_hessian,
                                        g4_vco_upper_hessian, g5_vco_lower_hessian, g5_vco_upper_hessian],
                           'updater': None}
    else:
        raise ValueError('Only 1 or 2 backstresses are supported in the tensile calibration at the moment.')
    # Make the point feasible
    if make_x0_feasible:
        x_0 = make_feasible(x_0, g_constants)
    vc_constraint_check(x_0, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup, rho_gamma_b_inf,
                        rho_gamma_b_sup)
    print(x_0)
    # Create the solver and run
    solver = sqp_factory('line-search', 'original', 'reciprocal', filtered_data, constraint_dict,
                         x_log_file, fun_log_file)
    # max_its for each Lagrangian step, maximum of 20 Lagrangian steps, and allow 2 full steps (of max_its)
    solver.set_maximum_iterations(max_its)
    solver.set_tolerance(tol)
    lambda_0 = np.zeros(np.shape(solver.constraint.get_g(x_0)))
    x = x_0.copy().reshape(-1, 1)
    [x_opt, lambda_opt, conv_criteria] = solver.solve_return_conv(x, lambda_0)
    # Check if the constraints were satisfied
    vc_constraint_check(x_opt, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup, rho_gamma_b_inf,
                        rho_gamma_b_sup)
    return [DummyResults(x_opt), None]


class DummyResults:
    """ Used to have a similar return type as the Scipy solver. """

    def __init__(self, x):
        self.x = x
        return


def make_feasible(x_0, c):
    """ Returns a feasible x with respect to the constraints g_3, g_4, g_5, and g_6.

    :param np.array x_0: (n,) Initial, infeasible point.
    :param dict c: Constants defining the constraints.
    :return np.array: (n,) Initial, feasible point.

    Notes:
        - x_0 should follow: x0 = [200000, 355, 1, 1, 1, 1] for one backstress
        - x_0 should follow: x0 = [200000, 355, 1, 1, 1, rho_gamma_12_inf, 1, 1] for two backstresses
    """
    n_backstresses = int((len(x_0) - 4) // 2)
    # Get the average of the bounds
    rho_yield_avg = 0.5 * (c['rho_yield_inf'] + c['rho_yield_sup'])
    rho_iso_avg = 0.5 * (c['rho_iso_inf'] + c['rho_iso_sup'])
    rho_gamma_avg = 0.5 * (c['rho_gamma_inf'] + c['rho_gamma_sup'])
    # Calculate the feasible set
    gamma2_0 = 1. / rho_gamma_avg * x_0[5]
    c1_0 = -x_0[5] * (-1. + rho_iso_avg) * (-1 + rho_yield_avg) * x_0[1]
    q_inf_0 = -c1_0 * rho_iso_avg / (x_0[5] * (-1. + rho_iso_avg))
    b_0 = 1. / rho_gamma_avg * x_0[5]
    if n_backstresses == 2:
        x_0[[2, 3, 4, 7]] = [q_inf_0, b_0, c1_0, gamma2_0]
    elif n_backstresses == 1:
        x_0[[2, 3, 4]] = [q_inf_0, b_0, c1_0]
    return x_0


def vc_constraint_check(x_opt, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup, rho_gamma_b_inf,
                        rho_gamma_b_sup):
    """ Checks if each of g_3, g_4, and g_5 were satisfied. """
    n_backstresses = int((len(x_opt) - 4) // 2)
    sum_c_gamma = 0.
    for i in range(n_backstresses):
        sum_c_gamma += x_opt[4 + 2 * i] / x_opt[5 + 2 * i]
    rho_yield_ratio = (x_opt[1] + x_opt[2] + sum_c_gamma) / x_opt[1]
    rho_iso_ratio = x_opt[2] / (x_opt[2] + sum_c_gamma)
    rho_gamma_ratio = x_opt[5] / x_opt[3]
    print('The rho_iso ratio is = {0}'.format(rho_iso_ratio))
    print('The rho_yield ratio is = {0}'.format(rho_yield_ratio))
    print('The rho_gamma_b ratio is = {0}'.format(rho_gamma_ratio))
    if not (rho_iso_inf <= rho_iso_ratio <= rho_iso_sup):
        print('Constraint on rho_iso violated!')
    if not (rho_yield_inf <= rho_yield_ratio <= rho_yield_sup):
        print('Constraint on rho_yield violated!')
    if not (rho_gamma_b_inf <= rho_gamma_ratio <= rho_gamma_b_sup):
        print('Constraint on rho_gamma violated!')
    return
