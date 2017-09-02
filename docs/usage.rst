=====
Usage
=====

To use RESSPyLab in a project::

    import RESSPyLab


The current version includes primarily functions to perform material parameter calibration of the Voce and Chaboche (VC) metal plasticity model.

There are three main useful functions in the library

.. code:: python

	def NTR_SVD_Solver(f,gradF,Hf,x_0): 
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

	def VCopt(x_0,listTests):
		# This function performs parameter optimization of the VC model for an ensemble of experimental data. 
		# 
		# listTests is a python list containg pandas dataframes of multiple "test"  
		return x_min

	


A working example using a Jupyter notebook_, along its data files can be found on gitHub_.

.. _notebook: https://nbviewer.jupyter.org/github/AlbanoCastroSousa/RESSPyLab/blob/master/VC_JupyterNotebook/RESSPyLab%20Parameter%20Calibration%20Orientation%20Notebook.ipynb

.. _gitHub: https://github.com/AlbanoCastroSousa/RESSPyLab/tree/master/VC_JupyterNotebook