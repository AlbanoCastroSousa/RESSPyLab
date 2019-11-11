"""@package multi_runner
Function to run several optimizations sequentially.
"""
from __future__ import print_function
import os
import errno
from .vc_parameter_identification import vc_param_opt
from .uvc_parameter_identification import uvc_param_opt
from .vc_limited_info_opt import vc_tensile_opt_scipy, vc_tensile_opt_auglag, vc_tensile_opt_linesearch


def dir_maker(directory):
    """ Makes directory if it doesn't exist, else does nothing. """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return


def opt_multi_run(data_dirs, output_dirs, data_names, should_filter, x_0s, model_type):
    """ Runs optimization for VC/UVC model multiple times sequentially.

    :param list data_dirs: [str] Paths to the data directories that contain the stress-strain data files.
    :param list output_dirs: [str] Paths to the directories that will contain the output files.
    :param list data_names: [str] Names for the data sets.
    :param list should_filter: [bool] If true, then apply filter to the data set, if False then do nothing.
    :param list x_0s: [np.ndarray] Starting point for each data sets.
    :param str model_type: 'VC' to use the Voce-Chaboche, 'UVC' to use the Updated Voce-Chaboche model.
    :return None: None

    Notes:
        - The data directories must only contain stress-strain data files.
        - This function creates each output directory if they do not already exist.
    """
    for i, d_dir in enumerate(data_dirs):
        o_dir = output_dirs[i]
        dir_maker(o_dir)
        name = data_names[i]
        x_log_file = os.path.join(o_dir, name + '_x_log.txt')
        fun_log_file = os.path.join(o_dir, name + '_fun_log.txt')
        file_list = [os.path.join(d_dir, p) for p in os.listdir(d_dir)]
        filt = should_filter[i]
        x_start = x_0s[i].copy()
        if model_type == 'VC':
            vc_param_opt(x_start, file_list, x_log_file, fun_log_file, filt)
        elif model_type == 'UVC':
            uvc_param_opt(x_start, file_list, x_log_file, fun_log_file, filter_data=filt)
        else:
            raise ValueError('model_type should be either VC or UVC')
    return


def tensile_opt_multi_run(data_files, output_dirs, data_names, should_filter, x_0s, model_type, constr_bounds,
                          feasible_start=True, algorithm='NITRO'):
    """ Runs the tensile test only optimization using the VC model multiple times sequentially.

    :param list data_files: [str] Paths to the files that contain the tensile stress-strain data.
    :param list output_dirs: [str] Paths to the directories that will contain the output files.
    :param list data_names: [str] Names for the data sets.
    :param list should_filter: [bool] If true, then apply filter to the data set, if False then do nothing.
    :param list x_0s: [np.ndarray] Starting point for each data sets.
    :param str model_type: 'VC' to use the Voce-Chaboche, 'UVC' to use the Updated Voce-Chaboche model.
    :param dict constr_bounds: Specifies the bounds on the constraints. See Notes for the keys/values.
    :param bool feasible_start: If True then modifies the starting point to be feasible, if False no modification.
    :param str algorithm: Specifies the optimization algorithm, see Notes for options.
    :return None: None

    Notes:
        - This function creates each output directory if they do not already exist.
        - constr_bounds contains the following key/value pairs (all values are float's):
            'rho_iso_inf': Lower bound on ratio of isotropic to total hardening at saturation.
            'rho_iso_sup': Upper bound on ratio of isotropic to total hardening at saturation.
            'rho_yield_inf': Lower bound on ratio of initial yield stress to total stress at saturation.
            'rho_yield_sup': Upper bound on ratio of initial yield stress to total stress at saturation.
            'rho_gamma_b_inf': Lower bound on ratio the rate of kinematic to isotropic hardening.
            'rho_gamma_b_sup': Upper bound on ratio the rate of kinematic to isotropic hardening.
            'rho_gamma_12_inf': Lower bound on ratio of gamma_1 to gamma_2.
            'rho_gamma_12_sup': Upper bound on ratio of gamma_1 to gamma_2.
        - algorithm options:
            'NITRO': NITRO algorithm
            'AugLag': augmented Lagrangian algorithm
            'LineSearch': SQP line-search algorithm
    """
    for i, d_path in enumerate(data_files):
        o_dir = output_dirs[i]
        dir_maker(o_dir)
        name = data_names[i]
        x_log_file = os.path.join(o_dir, name + '_li_x_log.txt')
        fun_log_file = os.path.join(o_dir, name + '_li_fun_log.txt')
        file_list = [d_path]
        filt = should_filter[i]
        x_start = x_0s[i].copy()
        cb = constr_bounds
        if algorithm == 'NITRO':
            opt_fun = vc_tensile_opt_scipy
        elif algorithm == 'AugLag':
            opt_fun = vc_tensile_opt_auglag
        elif algorithm == 'LineSearch':
            opt_fun = vc_tensile_opt_linesearch
        else:
            raise ValueError('Incorrect choice of algorithm parameter.')
        if model_type == 'VC':
            opt_fun(x_start, file_list,
                    cb['rho_iso_inf'], cb['rho_iso_sup'], cb['rho_yield_inf'], cb['rho_yield_sup'],
                    cb['rho_gamma_b_inf'], cb['rho_gamma_b_sup'],
                    cb['rho_gamma_12_inf'], cb['rho_gamma_12_sup'],
                    x_log_file, fun_log_file, filter_data=filt,
                    make_x0_feasible=feasible_start)
        else:
            raise ValueError('model_type should be VC')
    return
