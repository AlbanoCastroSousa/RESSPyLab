"""@package vcu_parameter_identification
Top level functions for parameter identification using the updated Voce-Chaboche model.
"""
from __future__ import print_function
import numpy as np

from scipy_constr_opt_factory import scipy_factory
from sqp_factory import sqp_factory
from auglag_factory import auglag_factory
from data_readers import load_and_filter_data_set, load_data_set
from rpl_constraint import RPLConstraint
from uvc_constraints import g1_constraint, g1_gradient, g1_hessian, g2_constraint, g2_gradient, g2_hessian, \
    positive_x_constraint, positive_x_gradient, positive_x_hessian


def uvc_param_opt(x_0, file_list, x_log_file='', function_log_file='', find_initial_point=True, filter_data=True,
                  regularization_lambda=None, step_iterations=(300, 1000, 3000), step_tolerances=(1.e-8, 1.e-2, 5.e-2)):
    """ Attempts to find a solution to the inverse problem of material identification for the updated Voce-Chaboche
    model.

    :param np.array x_0: (n, 1) Starting point for the optimization procedure.
    :param list file_list: [str] Paths to the data files to optimize the parameters over.
    :param str x_log_file: If not empty, then the model parameters are dumped to this file at each line-search
        iteration.
    :param str function_log_file: If not empty, then the iteration number, objective function value, and convergence
        criteria are dumped to this file at each line-search iteration.
    :param bool find_initial_point: If True then finds an initial point using an unconstrained optimization, if False
        then the user provides the initial (reasonably accurate) point.
    :param bool filter_data: If True then apply a filter to the data, else use the raw import.
    :param float regularization_lambda: Regularization parameter.
    :param list step_iterations: Number of iterations to use in each step of the solution procedure.
    :param list step_tolerances: Tolerance to use at each step of the solution procedure.
    :return list: The solution to the inverse problem.
        np.array x: Primal variables at solution point.
        np.array lagr: Dual variables at solution point.
    """
    # Initialize updated model parameters
    d_ini = 1.0
    a_ini = 200.0

    # Filter and set the data
    if filter_data:
        filtered_data = load_and_filter_data_set(file_list)
    else:
        filtered_data = load_data_set(file_list)

    # Get an initial point using unconstrained NTR optimization (original model), then use the specified D and a values
    if find_initial_point:
        constr_for_ntr = RPLConstraint({}, {}, [], [], [])
        ntr_solver = auglag_factory(filtered_data, x_dump_file='', fun_dump_file='', model_type='original',
                                    subproblem_solver='steihaug', barrier_type='reciprocal', accept_approx=False)
        ntr_solver.set_auglag_tol(1.e-8)
        x_original = x_0.copy()
        x_original = np.delete(x_original, [4, 5])
        x_ini = ntr_solver.augmented_lagrangian_opt(x_original.reshape(-1), constr_for_ntr)
        x_ini = np.insert(x_ini, 4, [d_ini, a_ini])
    else:
        x_ini = x_0.copy()

    # Start the constrained parameter search
    x = x_ini.copy()
    print("Starting constrained parameter search with {0:d} iterations at tol = {1:0.2e}".format(step_iterations[0],
                                                                                              step_tolerances[0]))
    [opt_results, dumper] = scipy_factory(x.reshape(-1), filtered_data, x_log_file, function_log_file,
                                          max_its=step_iterations[0], tol=step_tolerances[0],
                                          reg_lambda=regularization_lambda)
    for i in range(1, len(step_tolerances)):
        if opt_results.optimality > step_tolerances[i]:
            print("Continuing with tol = {1:0.2e} for {0:d} iterations.".format(step_iterations[i], step_tolerances[i]))
            [opt_results, dumper] = scipy_factory(opt_results.x, filtered_data, x_log_file, function_log_file,
                                                  max_its=step_iterations[i], tol=step_tolerances[i],
                                                  tr_radius=opt_results.tr_radius, dumper=dumper,
                                                  reg_lambda=regularization_lambda)
        else:
            break

    # Exit with the final parameters
    x = opt_results.x
    lagr = opt_results.v
    converge_criteria = opt_results.optimality
    print("Exiting with ||grad[L]|| = {0:e}".format(converge_criteria))
    print("x = {0}".format(x.reshape(-1)))
    print("dual_x = {0}".format(lagr))
    return [x, lagr]


def uvc_param_opt_ls(x_0, file_list, x_log_file='', function_log_file='', find_initial_point=True, filter_data=True):
    """ Attempts to find a solution to the inverse problem of material identification for the updated Voce-Chaboche
    model using the SQP line-search algorithm.

    :param np.array x_0: (n, 1) Starting point for the optimization procedure.
    :param list file_list: [str] Paths to the data files to optimize the parameters over.
    :param str x_log_file: If not empty, then the model parameters are dumped to this file at each line-search
        iteration.
    :param str function_log_file: If not empty, then the iteration number, objective function value, and convergence
        criteria are dumped to this file at each line-search iteration.
    :param bool find_initial_point: If True then finds an initial point using an unconstrained optimization, if False
        then the user provides the initial (reasonably accurate) point.
    :param bool filter_data: If True then apply a filter to the data, else use the raw import.
    :return list: The solution to the inverse problem.
        np.array x: Primal variables at solution point.
        np.array lagr: Dual variables at solution point.
    """
    # Initialize some parameters
    d_ini = 1.0
    a_ini = 200.0

    # Filter and set the data
    if filter_data:
        final_data = load_and_filter_data_set(file_list)
    else:
        final_data = load_data_set(file_list)

    # Set constraints
    constr_for_ntr = RPLConstraint({}, {}, [], [], [])

    # Set-up all the solvers
    ntr_solver = auglag_factory(final_data, x_dump_file='', fun_dump_file='', model_type='original',
                                subproblem_solver='steihaug', barrier_type='reciprocal', accept_approx=False)
    ntr_solver.set_auglag_tol(1.e-8)

    # Get an initial point using unconstrained NTR optimization (original model), then use the specified D and a values
    if find_initial_point:
        x_original = x_0 * 1.0
        x_original = np.delete(x_original, [4, 5])
        x_ini = ntr_solver.augmented_lagrangian_opt(x_original.reshape(-1), constr_for_ntr)
        x_ini = np.insert(x_ini, 4, [d_ini, a_ini])
    else:
        x_ini = x_0 * 1.0

    # Set the constraint dictionary
    constr_dict = {'constants': {'min_x': 0.1}, 'variables': {}, 'updater': None,
                   'functions': [g1_constraint, g2_constraint, positive_x_constraint],
                   'gradients': [g1_gradient, g2_gradient, positive_x_gradient],
                   'hessians': [g1_hessian, g2_hessian, positive_x_hessian]}
    # Set initial point
    sqp_solver = sqp_factory('line-search', 'updated', 'none', final_data, constr_dict, x_log_file, function_log_file)
    lambda_0 = np.zeros(np.shape(sqp_solver.constraint.get_g(x_0)))
    x = x_ini * 1.0

    # Start the constrained parameter search
    print("Starting constrained parameter search with 300 iterations at tol = 1.e-8")
    # Try 300 iterations with a strict tolerance
    sqp_solver.set_maximum_iterations(300)
    sqp_solver.set_tolerance(1.e-8)
    [x_sol, lambda_sol, conv_criteria] = sqp_solver.solve_return_conv(x, lambda_0)
    print("Continuing with tol = 1.e-2 for 1000 iterations.")
    if conv_criteria > 1.e-2:
        # Try 1000 iterations with a reduced tolerance if convergence isn't reached
        sqp_solver.reset_solver()
        sqp_solver.set_maximum_iterations(1000)
        sqp_solver.set_tolerance(1.e-2)
        [x_sol, lambda_sol, conv_criteria] = sqp_solver.solve_return_conv(x_sol, lambda_sol)
    print("Continuing with tol = 1.e-1 for 3000 iterations.")
    if conv_criteria > 1.e-1:
        # Further relax the tolerance for 3000 iterations
        sqp_solver.reset_solver()
        sqp_solver.set_maximum_iterations(3000)
        sqp_solver.set_tolerance(1.e-1)
        [x_sol, lambda_sol, conv_criteria] = sqp_solver.solve_return_conv(x_sol, lambda_sol)

    # Exit with the final parameters
    x = x_sol
    lagr = lambda_sol
    return [x, lagr]
