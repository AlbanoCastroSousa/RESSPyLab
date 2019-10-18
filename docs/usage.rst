=====
Usage
=====

To use RESSPyLab in a project::

    import RESSPyLab



Material Model Calibration
--------------------------

The current version of RESSPyLab includes functions for the parameter calibration of the original Voce-Chaboche and Updated Voce-Chaboche (UVC) material models.
The main functions useful for calibration are:

.. code:: python
    
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

    def uvc_param_opt(x_0, file_list, x_log_file='', fxn_log_file='', find_initial_point=True, filter_data=True,
                  step_iterations=(300, 1000, 3000), step_tolerances=(1.e-8, 1.e-2, 5.e-2)):
    """ Returns the best-fit parameters to the Updated Voce-Chaboche (UVC) model for a given set of stress-strain data.

    :param np.array x_0: [float, size n] Starting point for the optimization procedure.
    :param list file_list: [str] Paths to the data files to use in the optimization procedure.
    :param str x_log_file: File to track x values at each increment, if empty then don't track.
    :param str fxn_log_file: File to track objective function values at each increment, if empty then don't track.
    :param bool find_initial_point: If True then finds an initial point using an unconstrained optimization, if False
        then the user provides the initial point to the UVC model.
    :param bool filter_data: If True then apply a filter to the data, else use the raw import.
    :param list step_iterations: [int, size 3] Number of iterations to use in each step of the solution procedure.
    :param list step_tolerances: [float, size 3] Tolerance to use at each step of the solution procedure.
    :return list:
        np.array x: [float, size n] Solution point.
        np.array lagr: [float, size m] Dual variables (Lagrange multipliers) at solution point.
    """

Several other functions are provided for visualizations and post-processing.
Examples demonstrating the use of the calibration functions, and those for visualizations, are provided on github_ .

The current examples include:

- Voce-Chaboche model calibration with multiple stress-strain data: VC_Calibration_
- UVC model calibration: UVC_Calibration_
- Creating tables for parameters and hardening metrics: Post_Example_
- Visualizing the solution point: Vis_Example_


Version 0.1.5 Usage
-------------------

The remainder of this documentation is for the (old) 0.1.5 version of RESSPyLab.
This old version primarily includes functions to perform material parameter calibration of the Voce and Chaboche (VC) metal plasticity model.

There are five main useful functions in the library

.. code:: python

	def NTR_SVD_Solver(f,gradF,Hf,x_0): 
		# Newton-Trust Region solver with SVD preconditioning
		# f - function taking an array "x" of n floats as an argument and returning the real value of f
		# gradF - function taking an array "x" of n floats as an argument and returning the gradient of f, an array sized n
		# Hf - function taking an array "x" of n floats as an argument and returning Hessian of f, array sized n by n
		# x_0 - initial starting point; array sized n
		# x_min - local minimum of f; array sized n
		#...
		return x_min

	def NTR_J_Solver(f,gradF,Hf,x_0):
		# Newton-Trust Region solver with Jacobi preconditioning 
		# f - function taking an array "x" of n floats as an argument and returning the real value of f
		# gradF - function taking an array "x" of n floats as an argument and returning the gradient of f, an array sized n
		# Hf - function taking an array "x" of n floats as an argument and returning Hessian of f, array sized n by n
		# x_0 - initial starting point; array sized n
		# x_min - local minimum of f; array sized n
		#...
		return x_min

	def VCcurve(x,test):
		# x - set of parameters for the Voce and Chaboche material model
		# The order in x should be the following:
		# x=[E, sy0, Qinf, b, C_1, gamma_1, C_2, gamma_2, ..., ..., C_k, gamma_k]
		#
		# test is a pandas dataframe with columns named 'e_true', for the true strain and 'Sigma_true', for the true stress
		# simCurve is a pandas dataframe containing the simulated curve with the VC model. Integration is conducted with the discretization in "test"
		#...
		return simCurve

	def VCopt_SVD(x_0,listTests):
		# This function performs parameter optimization of the VC model for an ensemble of experimental data with SVD preconditioning. 
		# 
		# listTests is a python list containg pandas dataframes of multiple "test"  
		return x_min

	def VCopt_J(x_0,listTests):
		# This function performs parameter optimization of the VC model for an ensemble of experimental data with Jacobi preconditioning. 
		# 
		# listTests is a python list containg pandas dataframes of multiple "test"  
		return x_min

	


A working example for the 0.1.5 version using a Jupyter notebook_, along its data files can be found on github_.

.. _notebook: https://nbviewer.jupyter.org/github/AlbanoCastroSousa/RESSPyLab/blob/master/examples/Old_RESSPyLab_Parameter_Calibration_Orientation_Notebook.ipynb

.. _github: https://github.com/AlbanoCastroSousa/RESSPyLab/tree/master/examples/

.. _VC_Calibration: https://nbviewer.jupyter.org/github/AlbanoCastroSousa/RESSPyLab/blob/master/examples/VC_Calibration_Example_1.ipynb
.. _UVC_Calibration: https://nbviewer.jupyter.org/github/AlbanoCastroSousa/RESSPyLab/blob/master/examples/UVC_Calibration_Example_1.ipynb
.. _Post_Example: https://nbviewer.jupyter.org/github/AlbanoCastroSousa/RESSPyLab/blob/master/examples/Post_Processing_Example_1.ipynb
.. _Vis_Example: https://nbviewer.jupyter.org/github/AlbanoCastroSousa/RESSPyLab/blob/master/examples/Visualizations_Example_1.ipynb