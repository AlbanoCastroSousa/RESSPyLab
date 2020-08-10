"""@package vcu_plotter
Plot optimization results and test data for the updated Voce-Chaboche model.
"""
from uvc_model import sim_curve_uvc
from mpl_import import *


def uvc_data_plotter(x, test_data, output_dir, file_name, plot_label):
    """ Creates plots of updated Voce-Chaboche model overlayed on the test data.

    :param np.array x: Input parameters to the updated Voce-Chaboche model.
    :param list test_data: Collection of test data to plot.
    :param str output_dir: Directory to place files, if empty string then just displays the plots.
    :param str file_name: Identifier in output files.
    :param str plot_label: Name of Voce-Chaboche plot in the legend.
    :return list: Handles to each of the created plots.

    - Outputs in .pdf format
    - Appends i = [0, 1, ..., n-1] to the file name, where n is the number of test data sets included
    """
    handles = []
    for i, test in enumerate(test_data):
        sim_curve_upd = sim_curve_uvc(x, test)
        h = plt.figure()
        plt.plot(test['e_true'], test['Sigma_true'], c='k', label='Test', lw=0.75)
        plt.plot(sim_curve_upd['e_true'], sim_curve_upd['Sigma_true'], c='r', label=plot_label, lw=0.55)

        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin * 1.25, ymax)
        plt.legend(loc='lower right', ncol=len(x) + 1, frameon=False, columnspacing=0.5, mode=None, borderpad=0.,
                   borderaxespad=0.2, handlelength=1.25, handletextpad=0.3, fontsize='x-small')
        plt.xlabel(r'True Strain, $\varepsilon$')
        plt.ylabel(r'True Stress, $\sigma$ [MPa]')
        plt.tight_layout()
        if output_dir == '':
            plt.show()
        else:
            plt.savefig(output_dir + file_name + '_' + str(i) + '.pdf')
            plt.close()

        handles.append(h)
    return handles


def uvc_data_multi_plotter(x, test_data, output_dir, file_name, plot_labels, colors, styles, test_color='k'):
    """ Creates plots of updated Voce-Chaboche model overlayed on the test data for multiple parameter sets.

    :param list x: (np.array) Input parameters to the updated Voce-Chaboche model.
    :param list test_data: Collection of test data to plot.
    :param str output_dir: Directory to place files, if empty string then just displays the plots.
    :param str file_name: Identifier in output files.
    :param list plot_labels: (str) Names of the non test data lines in the legend.
    :param list colors: (str) Colors for each line.
    :param list styles: (str) Line style for each line.
    :param str test_color: Color for the test data line.
    :return list: Handles to each of the created plots.

    - Outputs in .pdf format
    - Appends i = [0, 1, ..., n-1] to the file name, where n is the number of test data sets included
    """
    handles = []
    for i, test in enumerate(test_data):
        h = plt.figure()
        plt.plot(test['e_true'], test['Sigma_true'], c=test_color, label='Test', lw=0.75)
        # Plot for all the sets of parameters
        for j, xj in enumerate(x):
            sim_curve_upd = sim_curve_uvc(xj, test)
            plt.plot(sim_curve_upd['e_true'], sim_curve_upd['Sigma_true'], c=colors[j], label=plot_labels[j], lw=0.55,
                     ls=styles[j])

        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        if len(plot_labels) < 3:
            ncol = len(x) + 1
            ax.set_ylim(ymin * 1.25, ymax)
        elif len(plot_labels) == 3:
            ncol = 2
            ax.set_ylim(ymin * 1.5, ymax)
        else:
            raise ValueError('More than 4 plots not supported at the moment.')

        plt.legend(loc='lower right', ncol=ncol, frameon=False, columnspacing=0.5, mode=None, borderpad=0.,
                   borderaxespad=0.2, handlelength=1.25, handletextpad=0.3, labelspacing=0.15, fontsize='x-small')
        plt.xlabel(r'True Strain, $\varepsilon$')
        plt.ylabel(r'True Stress, $\sigma$ [MPa]')
        plt.tight_layout()
        if output_dir == '':
            plt.show()
        else:
            plt.savefig(output_dir + file_name + '_' + str(i) + '.pdf')
            plt.close()

        handles.append(h)
    return handles
