"""@package multi_runner
Function to run several optimizations sequentially.
"""
from __future__ import print_function
import os
import errno
from .vc_parameter_identification import vc_param_opt
from .uvc_parameter_identification import uvc_param_opt


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
