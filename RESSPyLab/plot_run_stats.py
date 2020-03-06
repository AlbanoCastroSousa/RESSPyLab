"""@package plot_run_stats
Function to compare objective function and norm grad L between analyses.
"""
#import matplotlib.pyplot as plt
from mpl_import import *
import pandas as pd
import numpy as np


def multi_plot_obj_grad_lag(fun_log_files, plot_names=None, file_names=None, legend_both=False, log_both=False):
    """ Plots the value of the objective function and norm of the grad Lagrangian.

    :param list fun_log_files: [str] (m, ) Paths to the optimization log files.
    :param list plot_names: (m, ) Labels for the plot legends, if None then no legend is added to the figures.
    :param list file_names: (2, ) Full file path to save the figures, if None then doesn't save and just displays.
    :param bool legend_both: If False only puts a legend on the objective function plot, True legends both plots.
    :param bool log_both: If False uses log scale for the gradient plot, True uses log scale for both plots.
    :return list: (2, ) Figure handles for the objective function, norm grad Lagrangian plots.

    - The log files should be comma delimited and must contain the columns: iteration, function, norm_grad_Lagr
    - Only up to 4 log files are supported at the moment (i.e., m <= 4)
    - Your local plt.rcParams will govern much of the overall appearance of the plots
    - The file names are for the objective function, and the norm grad Lagrangian plots, respectively
    """
    # Line styles
    ls = ['-', '--', '-.', ':']
    color = '0.15'
    # Set-up plot axes
    obj_fun_fig, obj_fun_ax = plt.subplots()
    obj_fun_ax.set_xlabel('Iteration')
    obj_fun_ax.set_ylabel(r'$f(\mathbf{x})$')
    if log_both:
        obj_fun_ax.set_yscale('log')

    norm_grad_fig, norm_grad_ax = plt.subplots()
    norm_grad_ax.set_xlabel('Iteration')
    norm_grad_ax.set_ylabel(r'$\Vert \nabla\!_x \,L \Vert$')
    norm_grad_ax.set_yscale('log')

    # Plot each of the files
    for i, f in enumerate(fun_log_files):
        data = pd.read_csv(f, skipinitialspace=True)
        if plot_names is None:
            obj_fun_ax.plot(data['iteration'], data['function'], color, ls=ls[i])
            norm_grad_ax.plot(data['iteration'], data['norm_grad_Lagr'], color, ls=ls[i])
        else:
            obj_fun_ax.plot(data['iteration'], data['function'], color, ls=ls[i], label=plot_names[i])
            norm_grad_ax.plot(data['iteration'], data['norm_grad_Lagr'], color, ls=ls[i], label=plot_names[i])

    # Finalize the plots
    if plot_names is not None:
        obj_fun_ax.legend(loc='upper right')
        if legend_both:
            norm_grad_ax.legend(loc='upper right')
    obj_fun_fig.tight_layout()
    norm_grad_fig.tight_layout()

    # Show or save the figures
    if file_names is None:
        plt.show()
    else:
        obj_fun_fig.savefig(file_names[0])
        norm_grad_fig.savefig(file_names[1])
    return [obj_fun_fig, norm_grad_fig]


def multi_plot_x_values(x_log_files, plot_names=None, file_name=None, model_type='VC'):
    """ Plots the values of the optimization variables.

    :param list x_log_files: [str] (m, ) Paths to the optimization log files for the variable values.
    :param list plot_names: (m, ) Labels for the plot legends, if None then no legend is added to the figures.
    :param str file_name: Full file path to save the figure, if None then doesn't save and just displays.
    :param str model_type: 'VC' for Voce-Chaboche model, 'UVC' for the updated Voce-Chaboche model.
    :return plt.figure: Figure handle for plot.

    - The log files should be space delimited and follow the convention of parameters for the VC/UVC models
    - Only up to 4 log files are supported at the moment (i.e., m <= 4)
    - Your local plt.rcParams will govern much of the overall appearance of the plots
    """
    # Load a single value to get the number of backstresses
    x_test = np.loadtxt(x_log_files[0])[-1]
    if model_type == 'VC':
        n_back = (len(x_test) - 4) // 2
        basic_x_rows = 2
        # Parameter Name List
        y_names = [r'$E$ [MPa]', r'$\sigma_{y,0}$ [MPa]', r'$Q_{\infty}$ [MPa]', r'$b$']
    else:
        n_back = (len(x_test) - 6) // 2
        basic_x_rows = 3
        # Parameter Name List
        y_names = [r'$E$ [MPa]', r'$\sigma_{y,0}$ [MPa]', r'$Q_{\infty}$ [MPa]', r'$b$', r'$D_{\infty}$ [MPa]', r'$a$']
    for i in range(n_back):
        y_names.append(r'$C_{0}$ [MPa]'.format(i + 1))
        y_names.append(r'$\gamma_{0}$'.format(i + 1))

    # Line styles
    ls = ['-', '--', '-.', ':']
    color = '0.15'
    # Set-up plot axes
    fig_h = 1.5 * (basic_x_rows + n_back)
    fig = plt.figure(figsize=(6., fig_h))
    axes = []
    for i, _ in enumerate(y_names):
        axes.append(plt.subplot(basic_x_rows + n_back, 2, i + 1))

    # Plot each of the files
    for op_j, f in enumerate(x_log_files):
        param_vals = np.loadtxt(f)
        iterations = np.arange(1, len(param_vals) + 1)
        for i, ax in enumerate(axes):
            if plot_names is None:
                ax.plot(iterations, param_vals[:, i], color, ls=ls[i])
            else:
                ax.plot(iterations, param_vals[:, i], color, ls=ls[op_j], label=plot_names[op_j])
            ax.set_ylabel(y_names[i])

    # Finalize the plots
    if plot_names is not None:
        axes[0].legend(loc='best')
    axes[-2].set_xlabel('Iteration')
    axes[-1].set_xlabel('Iteration')
    fig.tight_layout()

    # Show or save the figures
    if file_name is None:
        plt.show()
    else:
        fig.savefig(file_name)
    return fig
