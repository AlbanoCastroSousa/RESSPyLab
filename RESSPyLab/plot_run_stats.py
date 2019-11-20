"""@package plot_run_stats
Function to compare objective function and norm grad L between analyses.
"""
import matplotlib.pyplot as plt
import pandas as pd


def plot_run_stats(fun_log_files, plot_names=None, file_names=None, legend_both=False, log_both=False):
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
