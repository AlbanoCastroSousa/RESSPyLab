"""@package mat_model_error_nda
Contains the objective function and its derivatives for optimization problems.
"""
from numdifftools import nd_algopy as nda
from barrier_function import BarrierFunction


class MatModelErrorNda:

    def __init__(self, material_model, test_data, barrier_function=None, use_cols=True, reg_qd_lambda=None):
        """ The error function for the material model, and gradients / Hessian of it.

        :param function material_model: A material model definition.
        :param list test_data: pandas.DataFrame's of test data that the error is calculated over.
        :param BarrierFunction barrier_function: Adds a barrier function to the objective function, optional.
        :param float reg_qd_lambda: (optional) If not None, applies a regularization to the Q_\infty and D_\infty
                                    parameters. Can be float or None.

        - material_model = f(x, test_data), where x are an numpy.array of the model parameters, here x is assumed to
            have shape=(n, )
        - The test_data DataFrame's must contain a column of 'e_true' and 'Sigma_true'
        - The gradient and Hessian of the objective function are calculated using algorithmic differentiation,
            therefore, the specified function should be amenable to algorithmic differentiation.
        - The regularization term is f_R(x) = reg_qd_lambda / 2. * ((Q_\infty)^2 + (D_\infty)^2)
        """
        self.error_fun = material_model
        self.test_data = test_data
        self.use_cols = use_cols
        if reg_qd_lambda is not None:
            self.reg_lambda = reg_qd_lambda / 2.
        else:
            self.reg_lambda = reg_qd_lambda

        # Barrier function initialization
        if barrier_function is None:
            self.use_barrier = False
        elif isinstance(barrier_function, BarrierFunction):
            self.use_barrier = True
            self.barrier = barrier_function
        else:
            raise RuntimeError("Improper barrier function specified.")

        # Gradient and Hessian definitions, this must always be after the barrier function is defined
        # The grad/hess are defined here so that nda is not set-up multiple times
        self.__grad_error_fun = nda.Gradient(self.value)
        self.__hess_error_fun = nda.Hessian(self.value)
        return

    def value(self, x):
        """ Returns the value of the objective function. """
        x1 = x.reshape(-1)  # to make compatible with the form assumed in the material model

        # Compute the new value of f
        error_ensemble = 0.
        for cleanedTest in self.test_data:
            error_ensemble = error_ensemble + self.error_fun(x1, cleanedTest)
        # Barrier function definition
        if self.use_barrier:
            for xi in x1:
                if xi > 0.:
                    # Don't apply barrier if < 0 so that can let back across if gets to the other side
                    error_ensemble = error_ensemble + self.barrier.value(xi)
        # Add the regularization term
        if self.reg_lambda is not None:
            error_ensemble += self.regularization(x)

        return error_ensemble

    def grad(self, x):
        """ Returns the gradient of the objective function. """
        x1 = x.reshape(-1)  # to make compatible with the form assumed in the material model
        if self.use_cols:
            return self.__grad_error_fun(x1).reshape(len(x), 1)
        else:
            return self.__grad_error_fun(x1)

    def hess(self, x):
        """ Returns the Hessian of the objective function. """
        x1 = x.reshape(-1)  # to make compatible with the form assumed in the material model
        return self.__hess_error_fun(x1)

    def regularization(self, x):
        """ L_2 regularization applied to Q_\infty and D_\infty. """
        # Assumes that Q_\infty is x[2], and D_\infty is x[4]
        return self.reg_lambda * (x[2] ** 2 + x[4] ** 2)
