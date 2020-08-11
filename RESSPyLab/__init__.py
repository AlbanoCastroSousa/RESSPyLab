# -*- coding: utf-8 -*-

"""Top-level package for RESSPyLab."""

__author__ = """Albano de Castro e Sousa and Alexander R. Hartloper"""
__email__ = 'albano.sousa@epfl.ch'
__version__ = '1.1.2'

# From the base version
from .RESSPyLab import errorTest_scl
from .RESSPyLab import errorEnsemble_nda
from .RESSPyLab import steihaug
from .RESSPyLab import NTR_SVD_Solver
from .RESSPyLab import VCopt_SVD
from .RESSPyLab import NTR_J_Solver
from .RESSPyLab import VCopt_J
from .RESSPyLab import VCsimCurve
from .RESSPyLab import Lagrangian
from .RESSPyLab import gradLag
from .RESSPyLab import hessLag
from .RESSPyLab import makeG
from .RESSPyLab import NTR_J_Solver_Lag
from .RESSPyLab import AugLag_Opt

# For the original Voce-Chaboche model with no constraints
from .vc_parameter_identification import vc_param_opt, vc_consistency_metric

# For the updated Voce-Chaboche model
from .data_readers import load_and_filter_data_set, load_data_set
from .auglag_factory import auglag_factory, constrained_auglag_opt
from .uvc_model import sim_curve_uvc, calc_phi_total, uvc_consistency_metric
from .sqp_factory import sqp_factory
from .uvc_parameter_identification import uvc_param_opt, uvc_param_opt_ls

# For tensile only optimization
from .vc_limited_info_opt import vc_tensile_opt_scipy

# For running several optimizations sequentially
from .multi_runner import opt_multi_run, tensile_opt_multi_run

# For output
from .summary_tables_maker import summary_tables_maker_uvc, summary_tables_maker_vc
from .solution_vis import solution_visualizations
from .uvc_plotter import uvc_data_plotter, uvc_data_multi_plotter
from .plot_run_stats import multi_plot_obj_grad_lag, multi_plot_x_values
