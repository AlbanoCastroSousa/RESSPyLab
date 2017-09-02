=====
Usage
=====

To use RESSPyLab in a project::

    import RESSPyLab

The current version includes primarily functions to perform material parameter calibration of the Voce and Chaboche metal plasticity model.

There are three main useful functions in the library::

	NTR_SVD_Solver(f,gradF,Hf,x_0) -> x_min

	VCopt(x_0,listTests) -> x_min

	VCcurve(x,test) -> simCurve