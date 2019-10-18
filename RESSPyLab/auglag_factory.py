"""@package auglag_factory
Functions to build augmented Lagrangian optimization problems.
"""
from rpl_constraint import RPLConstraint
from uvc_model import error_single_test_uvc
from RESSPyLab import errorTest_scl
from log_barrier import LogBarrier
from reciprocal_barrier import ReciprocalBarrier
from auglag_gen_solver import AugLagGenSolver
from steihaug_col import steihaug_col
from model_minimizer import model_minimizer
from basic_dumper import BasicDumper

import numpy as np


def auglag_factory(data, x_dump_file, fun_dump_file, model_type, subproblem_solver, barrier_type, accept_approx):
    """ Returns an initialized AugLagGenSolver.

    :param list data: Test data to use in optimization.
    :param str x_dump_file: File to dump x output.
    :param str fun_dump_file: File to dump iteration number, function, and convergence criteria output.
    :param str model_type: 'updated' for the updated model, 'original' for the original model
    :param str subproblem_solver: 'steihaug' for Steihaug-Toint solver, or 'model-min' for the model minimizer.
    :param str barrier_type: 'log' for log-barrier function, 'reciprocal' for reciprocal barrier function, or 'none' if
        no barrier is to be used.
    :param bool accept_approx: Use secondary criteria or not.
    :return AugLagGenSolver: solver object.
    """
    use_string = '\nUsage: auglag_factory(data, x_dump_file, fun_dump_file, model_type, subproblem_solver,' \
                 'barrier_type, accept_approx)'

    if model_type == 'updated':
        f = error_single_test_uvc
    elif model_type == 'original':
        f = errorTest_scl
    else:
        raise ValueError('Invalid model specification.' + use_string)

    if subproblem_solver == 'steihaug':
        s = steihaug_col
    elif subproblem_solver == 'model-min':
        s = model_minimizer
    else:
        raise ValueError('Invalid subproblem solver.' + use_string)

    if barrier_type == 'log':
        b = LogBarrier()
    elif barrier_type == 'reciprocal':
        b = ReciprocalBarrier()
    elif barrier_type == 'none':
        b = None
    else:
        print 'Warning: invalid barrier type, proceeding without a barrier!'
        b = None

    d = BasicDumper(output_file=x_dump_file, numpy_printopts={'precision': 3, 'suppress': True}, verbose_dump_freq=1,
                    function_output_file=fun_dump_file)

    solver = AugLagGenSolver(data, f, s, dumper=d, barrier_function=b, accept_approx_its=accept_approx)
    return solver


def constrained_auglag_opt(x_0, constraints, solver):
    """ Runs the optimization using the specified solver and constraints.

    :param list x_0: Initial model parameters.
    :param dict constraints: Functions to generate the constraints, gradients, and hessians; constants and variables
        used by these functions.
    :param AugLagGenSolver solver: Properly initialized solver.
    :return np.array:

    - constraints has keys: 'constants', 'variables', 'functions', 'gradients', 'hessians', 'updater'. The values
        correspond to those in the AugLagConstraint class.
    """
    x = np.array(x_0) * 1.0
    c = RPLConstraint(constraints['constants'], constraints['variables'], constraints['functions'],
                      constraints['gradients'], constraints['hessians'], constraints['updater'])
    x_sol = solver.augmented_lagrangian_opt(x, c)
    return x_sol
