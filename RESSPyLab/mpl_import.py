import matplotlib as mpl
import matplotlib.pyplot as plt

MPL_LINE_WIDTH = 0.55
MPL_FONT_SIZE = 9.0
MPL_LEG_FONT_SIZE = 8.0

mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.25
mpl.rcParams['xtick.minor.width'] = 0.25
mpl.rcParams['xtick.labelsize'] = MPL_LEG_FONT_SIZE
mpl.rcParams['ytick.major.width'] = 0.25
mpl.rcParams['ytick.minor.width'] = 0.25
mpl.rcParams['ytick.labelsize'] = MPL_LEG_FONT_SIZE

plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': 'Computer Modern Roman', 'size': MPL_FONT_SIZE})
plt.rc('lines', **{'linewidth': MPL_LINE_WIDTH})
plt.rc('axes', **{'labelsize': MPL_FONT_SIZE})
plt.rc('legend', **{'frameon': False, 'fontsize': MPL_LEG_FONT_SIZE})


def cm2inch(value):
    """ function for resizing figures to page dimensions """
    return value / 2.54
