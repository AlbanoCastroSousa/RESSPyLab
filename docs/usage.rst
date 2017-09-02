=====
Usage
=====

To use RESSPyLab in a project::

    import RESSPyLab


The current version includes primarily functions to perform material parameter calibration of the Voce and Chaboche metal plasticity model.

There are three main useful functions in the library

.. code:: python

	def NTR_SVD_Solver(f,gradF,Hf,x_0): 
		# f - python function taking an numpy array "x" of n variables and returning a real value
		# gradF - gradient of f, array size n
		# Hf - Hessian of f, array sized n by n
		# x_min - local minimum of f
		return x_min

	def VCopt(x_0,listTests):  
		return x_min

	def VCcurve(x,test): 
		return simCurve