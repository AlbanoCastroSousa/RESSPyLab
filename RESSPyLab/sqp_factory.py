"""@package sqp_factory
Functions to easily set-up analysis using the RESSPyLab SQP solvers.
"""
from log_barrier import LogBarrier
from reciprocal_barrier import ReciprocalBarrier
from RESSPyLab import errorTest_scl
from uvc_model import error_single_test_uvc
from basic_dumper import BasicDumper
from mat_model_error_nda import MatModelErrorNda
from rpl_constraint import RPLConstraint
from eq_from_ineq_constraint import EqFromIneqConstraint
from sqp_trustregion import SqpTrustregion
from sqp_linesearch import SqpLinesearch


def sqp_factory(sqp_solver_type, model_type, barrier_type, test_data, constraint_dict, dump_file='', function_file=''):
    """ Returns an initialized SQP solver with the properties specified.

    :param str sqp_solver_type: line-search or trust-region
    :param str model_type: original or updated
    :param str barrier_type: none, reciprocal, or log
    :param list test_data: (pd.DataFrame) data to use
    :param dict constraint_dict: constraints
    :param str dump_file: where to dump the x values
    :param str function_file: where to dump the function values
    :return SqpSolver: The initialized SQP solver
    """
    use_string = '\nUsage: sqp_factory(sqp_solver_type, model_type, barrier_type, test_data, constraint_dict, ' \
                 'dump_file, function_file)'

    if model_type == 'updated':
        f = error_single_test_uvc
    elif model_type == 'original':
        f = errorTest_scl
    else:
        raise ValueError('Invalid model specification.' + use_string)

    if barrier_type == 'log':
        b = LogBarrier()
    elif barrier_type == 'reciprocal':
        b = ReciprocalBarrier()
    elif barrier_type == 'none':
        b = None
    else:
        print 'Warning: invalid barrier type, proceeding without a barrier!'
        b = None

    objective_function = MatModelErrorNda(f, test_data, b)
    d = BasicDumper(dump_file, numpy_printopts={'precision': 3, 'suppress': True}, verbose_dump_freq=1,
                    function_output_file=function_file)

    if sqp_solver_type == 'line-search':
        c = RPLConstraint(constraint_dict['constants'], constraint_dict['variables'], constraint_dict['functions'],
                          constraint_dict['gradients'], constraint_dict['hessians'], constraint_dict['updater'])
        solver = SqpLinesearch(objective_function, c, dumper=d)
    elif sqp_solver_type == 'trust-region':
        c = EqFromIneqConstraint(constraint_dict['constants'], constraint_dict['variables'],
                                 constraint_dict['functions'], constraint_dict['gradients'],
                                 constraint_dict['hessians'], constraint_dict['updater'])
        solver = SqpTrustregion(objective_function, c, dumper=d)
    else:
        raise ValueError('Invalid solver specification.' + use_string)

    return solver
