"""@package vcu_limited_info_opt
Functions for limited information optimization using the updated Voce-Chaboche model.
"""
import scipy.optimize as opt

from vc_limited_info_opt import GWrapper, DummyResults
from uvc_model import error_single_test_uvc
from mat_model_error_nda import MatModelErrorNda
from scipy_dumper import ScipyBasicDumper
from uvc_constraints import *
from uvc_li_opt_constraints import *
from scipy_constr_opt_factory import g1_scipy, grad1_scipy, hess1_scipy, g2_scipy, grad2_scipy, hess2_scipy
from auglag_factory import constrained_auglag_opt, auglag_factory
from .data_readers import load_and_filter_data_set, load_data_set


def uvc_tensile_opt_scipy(x_0, file_list, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup,
                          rho_gamma_b_inf, rho_gamma_b_sup, rho_gamma_12_inf, rho_gamma_12_sup,
                          rho_d_inf, rho_d_sup,
                          x_log_file='', fun_log_file='', filter_data=True,
                          max_its=600, tol=1.e-8, make_x0_feasible=True):
    """ Return parameters based on a single tensile test for the updated VC model using the trust-constr method.

    USE OF THIS METHOD IS STRONGLY DISCOURAGED, USE THE ORIGINAL MODEL.
    """
    # Load the data
    if filter_data:
        filtered_data = load_and_filter_data_set(file_list)
    else:
        filtered_data = load_data_set(file_list)

    # Define the objective function
    objective_function = MatModelErrorNda(error_single_test_uvc, filtered_data, use_cols=False)
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
                   'rho_gamma_b_inf': rho_gamma_b_inf, 'rho_gamma_b_sup': rho_gamma_b_sup,
                   'rho_gamma_12_inf': rho_gamma_12_inf, 'rho_gamma_12_sup': rho_gamma_12_sup,
                   'rho_d_inf': rho_d_inf, 'rho_d_sup': rho_d_sup}
    # Updated model hardening constraints
    g1 = opt.NonlinearConstraint(g1_scipy, -np.inf, 0., jac=grad1_scipy, hess=hess1_scipy)
    g2 = opt.NonlinearConstraint(g2_scipy, -np.inf, 0., jac=grad2_scipy, hess=hess2_scipy)
    # g_3
    g3_low = GWrapper(g3_lower, g3_lower_gradient, g3_lower_hessian, g_constants)
    g3_high = GWrapper(g3_upper, g3_upper_gradient, g3_upper_hessian, g_constants)
    g3_inf_constr = opt.NonlinearConstraint(g3_low.f, -np.inf, 0., jac=g3_low.gf, hess=g3_low.hf)
    g3_sup_constr = opt.NonlinearConstraint(g3_high.f, -np.inf, 0., jac=g3_high.gf, hess=g3_high.hf)
    # g_4
    g4_low = GWrapper(g4_lower, g4_lower_gradient, g4_lower_hessian, g_constants)
    g4_high = GWrapper(g4_upper, g4_upper_gradient, g4_upper_hessian, g_constants)
    g4_inf_constr = opt.NonlinearConstraint(g4_low.f, -np.inf, 0., jac=g4_low.gf, hess=g4_low.hf)
    g4_sup_constr = opt.NonlinearConstraint(g4_high.f, -np.inf, 0., jac=g4_high.gf, hess=g4_high.hf)
    # g_5
    g5_low = GWrapper(g5_lower, g5_lower_gradient, g5_lower_hessian, g_constants)
    g5_high = GWrapper(g5_upper, g5_upper_gradient, g5_upper_hessian, g_constants)
    g5_inf_constr = opt.NonlinearConstraint(g5_low.f, -np.inf, 0., jac=g5_low.gf, hess=g5_low.hf)
    g5_sup_constr = opt.NonlinearConstraint(g5_high.f, -np.inf, 0., jac=g5_high.gf, hess=g5_high.hf)
    # g_6
    g_6_low = GWrapper(g6_lower, g6_lower_gradient, g6_lower_hessian, g_constants)
    g_6_high = GWrapper(g6_upper, g6_upper_gradient, g6_upper_hessian, g_constants)
    g6_inf_constr = opt.NonlinearConstraint(g_6_low.f, -np.inf, 0., jac=g_6_low.gf, hess=g_6_low.hf)
    g6_sup_constr = opt.NonlinearConstraint(g_6_high.f, -np.inf, 0., jac=g_6_high.gf, hess=g_6_high.hf)
    # g_7
    g_7_low = GWrapper(g7_lower, g7_lower_gradient, g7_lower_hessian, g_constants)
    g_7_high = GWrapper(g7_upper, g7_upper_gradient, g7_upper_hessian, g_constants)
    g7_inf_constr = opt.NonlinearConstraint(g_7_low.f, -np.inf, 0., jac=g_7_low.gf, hess=g_7_low.hf)
    g7_sup_constr = opt.NonlinearConstraint(g_7_high.f, -np.inf, 0., jac=g_7_high.gf, hess=g_7_high.hf)

    # Make the tuple of constraints
    n_backstresses = int((len(x_0) - 6) / 2)
    if n_backstresses >= 2:
        constraints = (g1, g2, g3_inf_constr, g3_sup_constr, g4_inf_constr, g4_sup_constr, g5_inf_constr, g5_sup_constr,
                       g6_inf_constr, g6_sup_constr, g7_inf_constr, g7_sup_constr)
    elif n_backstresses == 1:
        constraints = (g1, g2, g3_inf_constr, g3_sup_constr, g4_inf_constr, g4_sup_constr, g5_inf_constr, g5_sup_constr,
                       g7_inf_constr, g7_sup_constr)

    # Create a new dumper if None is provided
    dumper = ScipyBasicDumper(x_log_file, fun_log_file)

    # Make the point feasible
    if make_x0_feasible:
        x_0 = make_feasible_uvc(x_0, g_constants)

    # Run the minimization
    scipy_sol = opt.minimize(fun, np.array(x_0), method='trust-constr', jac=jac, hess=hess, bounds=bounds,
                             constraints=constraints, callback=dumper.dump,
                             options={'maxiter': max_its, 'verbose': 2, 'gtol': tol})
    # Check if the constraints were satisfied
    uvc_constraint_check(scipy_sol.x, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup, rho_gamma_b_inf,
                         rho_gamma_b_sup, rho_d_inf, rho_d_sup)
    return [scipy_sol, dumper]


def uvc_limited_info_opt_auglag(x_0, data, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup,
                                rho_gamma_inf, rho_gamma_sup, rho_gamma_12_inf, rho_gamma_12_sup,
                                rho_d_inf, rho_d_sup,
                                x_log_file='', fun_log_file='',
                                max_its=200, tol=1.e-8, make_x0_feasible=True):
    """ Return parameters based on a single tensile test for the original VC model using the augmented Lagrangian
    method.

    USE OF THIS METHOD IS STRONGLY DISCOURAGED, USE THE ORIGINAL MODEL."""

    # The constants for the constraint functions
    g_constants = {'rho_yield_inf': rho_yield_inf, 'rho_yield_sup': rho_yield_sup,
                   'rho_iso_inf': rho_iso_inf, 'rho_iso_sup': rho_iso_sup,
                   'rho_gamma_inf': rho_gamma_inf, 'rho_gamma_sup': rho_gamma_sup,
                   'rho_gamma_12_inf': rho_gamma_12_inf, 'rho_gamma_12_sup': rho_gamma_12_sup,
                   'rho_d_inf': rho_d_inf, 'rho_d_sup': rho_d_sup}

    # Set-up constraints
    n_backstresses = int((len(x_0) - 6) / 2)
    if n_backstresses == 1:
        constraint_dict = {'constants': g_constants, 'variables': {},
                           'functions': [g1_constraint, g2_constraint,
                                         g3_lower, g3_upper, g4_lower, g4_upper,
                                         g5_lower, g5_upper, g7_lower, g7_upper],
                           'gradients': [g1_gradient, g2_gradient,
                                         g3_lower_gradient, g3_upper_gradient, g4_lower_gradient,
                                         g4_upper_gradient, g5_lower_gradient, g5_upper_gradient,
                                         g7_lower_gradient, g7_upper_gradient],
                           'hessians': [g1_hessian, g2_hessian,
                                        g3_lower_hessian, g3_upper_hessian, g4_lower_hessian,
                                        g4_upper_hessian, g5_lower_hessian, g5_upper_hessian,
                                        g7_lower_hessian, g7_upper_hessian],
                           'updater': None}
    else:
        constraint_dict = {'constants': g_constants, 'variables': {},
                           'functions': [g1_constraint, g2_constraint,
                                         g3_lower, g3_upper, g4_lower, g4_upper,
                                         g5_lower, g5_upper, g6_lower, g6_upper, g7_lower, g7_upper],
                           'gradients': [g1_gradient, g2_gradient,
                                         g3_lower_gradient, g3_upper_gradient, g4_lower_gradient,
                                         g4_upper_gradient, g5_lower_gradient, g5_upper_gradient,
                                         g6_lower_gradient, g6_upper_gradient, g7_lower_gradient, g7_upper_gradient],
                           'hessians': [g1_hessian, g2_hessian,
                                        g3_lower_hessian, g3_upper_hessian, g4_lower_hessian,
                                        g4_upper_hessian, g5_lower_hessian, g5_upper_hessian,
                                        g6_lower_hessian, g6_upper_hessian, g7_lower_hessian, g7_upper_hessian],
                           'updater': None}

    # Make the point feasible
    if make_x0_feasible:
        x_0 = make_feasible_uvc(x_0, g_constants)
    # Create the solver and run
    solver = auglag_factory(data, x_log_file, fun_log_file, 'updated', 'steihaug', 'reciprocal', False)
    x_opt = constrained_auglag_opt(x_0, constraint_dict, solver)
    return [DummyResults(x_opt), None]


def make_feasible_uvc(x_0, c):
    """ Returns a feasible x with respect to the constraints g_3, g_4, g_5, and g_6.

    :param np.array x_0: (n,) Initial, infeasible point.
    :param dict c: Constants defining the constraints.
    :return np.array: (n,) Initial, feasible point.
    """
    n_backstresses = int((len(x_0) - 6) / 2)
    # Get the average of the bounds
    rho_yield_avg = 0.5 * (c['rho_yield_inf'] + c['rho_yield_sup'])
    rho_iso_avg = 0.5 * (c['rho_iso_inf'] + c['rho_iso_sup'])
    rho_gamma_avg = 0.5 * (c['rho_gamma_b_inf'] + c['rho_gamma_b_sup'])
    rho_d_avg = 0.5 * (c['rho_d_inf'] + c['rho_d_sup'])
    # Calculate the feasible set (considering only 1 backstress)
    gamma2_0 = 1. / rho_gamma_avg * x_0[7]
    c1_0 = -x_0[7] * (-1. + rho_iso_avg) * (-1 + rho_yield_avg) * x_0[1]
    q_inf_0 = -c1_0 * rho_iso_avg / (x_0[7] * (-1. + rho_iso_avg))
    b_0 = 1. / rho_gamma_avg * x_0[7]
    d_0 = rho_d_avg * (q_inf_0 + c1_0 / x_0[7])
    if n_backstresses == 2:
        x_0[[2, 3, 4, 6, 9]] = [q_inf_0, b_0, d_0, c1_0, gamma2_0]
    elif n_backstresses == 1:
        x_0[[2, 3, 4, 6]] = [q_inf_0, b_0, d_0, c1_0]
    return x_0


def uvc_constraint_check(x_opt, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup, rho_gamma_b_inf,
                         rho_gamma_b_sup, rho_d_inf, rho_d_sup):
    """ Checks if each of g_3, g_4, and g_5 were satisfied. """
    n_backstresses = int((len(x_opt) - 6) // 2)
    sum_c_gamma = 0.
    for i in range(n_backstresses):
        sum_c_gamma += x_opt[6 + 2 * i] / x_opt[7 + 2 * i]
    rho_yield_ratio = (x_opt[1] + x_opt[2] + sum_c_gamma) / x_opt[1]
    rho_iso_ratio = x_opt[2] / (x_opt[2] + sum_c_gamma)
    rho_gamma_b_ratio = x_opt[7] / x_opt[3]
    rho_d_ratio = x_opt[4] / (x_opt[2] + sum_c_gamma)
    print('The rho_iso ratio is = {0}'.format(rho_iso_ratio))
    print('The rho_yield ratio is = {0}'.format(rho_yield_ratio))
    print('The rho_gamma_b ratio is = {0}'.format(rho_gamma_b_ratio))
    print('The rho_d ratio is = {0}'.format(rho_d_ratio))

    if not (rho_iso_inf <= rho_iso_ratio <= rho_iso_sup):
        print('Constraint on rho_iso violated!')
    if not (rho_yield_inf <= rho_yield_ratio <= rho_yield_sup):
        print('Constraint on rho_yield violated!')
    if not (rho_gamma_b_inf <= rho_gamma_b_ratio <= rho_gamma_b_sup):
        print('Constraint on rho_gamma violated!')
    if not (rho_d_inf <= rho_d_ratio <= rho_d_sup):
        print('Constraint on rho_d violated!')
    return
