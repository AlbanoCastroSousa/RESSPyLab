"""
Top level function for calibration of the original Voce-Chaboche model.
"""
import numpy as np
from numdifftools import nd_algopy as nda
from .auglag_factory import auglag_factory, constrained_auglag_opt
from .data_readers import load_and_filter_data_set, load_data_set
from .uvc_model import test_total_area
from .RESSPyLab import errorTest_scl


def vc_param_opt(x_0, file_list, x_log_file, fxn_log_file, filter_data=True):
    """ Returns the best-fit parameters to the Voce-Chaboche model for a given set of stress-strain data.

    :param list x_0: Initial solution point.
    :param list file_list: [str] Paths to the data files to use in the optimization procedure.
    :param str x_log_file: File to track x values at each increment, if empty then don't track.
    :param str fxn_log_file: File to track objective function values at each increment, if empty then don't track.
    :param bool filter_data: If True then apply a filter to the data, else use the raw import.
    :return np.array: Solution point.

    Notes:
    - This function uses the augmented Lagrangian method without any constraints, therefore the method reduces to the
    Newton trust-region method.
    """
    if filter_data:
        filtered_data = load_and_filter_data_set(file_list)
    else:
        filtered_data = load_data_set(file_list)

    solver = auglag_factory(filtered_data, x_log_file, fxn_log_file, 'original', 'steihaug', 'reciprocal', False)
    constraints = {'constants': [], 'variables': [], 'functions': [], 'gradients': [], 'hessians': [],
                   'updater': None}
    x_sol = constrained_auglag_opt(x_0, constraints, solver)
    return x_sol


def vc_get_hessian(x, data):
    """ Returns the Hessian of the material model error function for a given set of test data evaluated at x.

    :param np.array x: Original Voce-Chaboche material model parameters.
    :param list data: (pd.DataFrame) Stress-strain history for each test considered.
    :return np.array: Hessian matrix of the error function.
    """

    def f(xi):
        val = 0.
        for d in data:
            val += errorTest_scl(xi, d)
        return val

    hess_fun = nda.Hessian(f)
    return hess_fun(x)


def vc_consistency_metric(x_base, x_sample, data):
    """ Returns the xi_2 consistency metric from de Sousa and Lignos 2019 using the original Voce-Chaboche model.

    :param np.array x_base: Original Voce-Chaboche material model parameters from the base case.
    :param np.array x_sample: Original Voce-Chaboche material model parameters from the sample case.
    :param list data: (pd.DataFrame) Stress-strain history for each test considered.
    :return float: Increase in quadratic approximation from the base to the sample case.
    """
    x_diff = x_sample - x_base
    hess_base = vc_get_hessian(x_base, data)
    numerator = np.dot(x_diff, hess_base.dot(x_diff))
    # Total area is independent of the x choice, just need to run this
    denominator = test_total_area(np.insert(x_base, 4, [0., 1.]), data)
    return np.sqrt(numerator / denominator)
