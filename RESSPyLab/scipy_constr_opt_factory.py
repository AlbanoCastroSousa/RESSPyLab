"""@package scipy_constr_opt_factor
Functions to package the scipy optimization methods.
"""
import numpy as np
import scipy.optimize as opt

from uvc_model import error_single_test_uvc
from mat_model_error_nda import MatModelErrorNda
from scipy_dumper import ScipyBasicDumper
from uvc_constraints import g1_constraint, g1_gradient, g1_hessian, g2_constraint, g2_gradient, g2_hessian


# Set solver and run
def scipy_factory(x_0, filtered_data, x_log_file, fun_log_file, max_its=200, tol=1e-8, tr_radius=1., dumper=None,
                  min_x_bound=0.01, reg_lambda=None):
    """ Creates and runs a constrained optimization problem for the updated Voce-Chaboche model.

    :param np.array x_0: Initial primal point.
    :param list filtered_data: (pd.DataFrame) Test data to optimize over.
    :param str x_log_file: Path to file to write x history.
    :param str fun_log_file: Path to file to write objective function history.
    :param int max_its: Maximum iterations allowed.
    :param float tol: Exit tolerance on the norm of grad[L].
    :param float tr_radius: Initial trust-region radius.
    :param ScipyBasicDumper dumper: Dumper for the analysis.
    :param float min_x_bound: Minimum value allowable for primal variables.
    :param float reg_lambda: Constant for regularization.
    :return list:
        - (np.array): Final solution point.
        - (GenSolverDumper): Dumper object used in the analysis.

    Notes:
        - Use of the regularization parameter is strongly discouraged.
    """
    # Define the objective function
    objective_function = MatModelErrorNda(error_single_test_uvc, filtered_data, use_cols=False,
                                          reg_qd_lambda=reg_lambda)
    fun = objective_function.value
    jac = objective_function.grad
    hess = objective_function.hess

    # Set up the constraints for the optimization
    x_lb = [min_x_bound for i in range(len(x_0))]
    x_ub = [np.inf for i in range(len(x_0))]
    bounds = opt.Bounds(x_lb, x_ub, keep_feasible=True)
    g1 = opt.NonlinearConstraint(g1_scipy, -np.inf, 0., jac=grad1_scipy, hess=hess1_scipy)
    g2 = opt.NonlinearConstraint(g2_scipy, -np.inf, 0., jac=grad2_scipy, hess=hess2_scipy)
    constraints = (g1, g2)

    # Create a new dumper if None is provided
    if dumper is None:
        dumper = ScipyBasicDumper(x_log_file, fun_log_file)

    # Run the minimization
    scipy_sol = opt.minimize(fun, np.array(x_0), method='trust-constr', jac=jac, hess=hess, bounds=bounds,
                             constraints=constraints, callback=dumper.dump,
                             options={'maxiter': max_its, 'verbose': 2, 'gtol': tol, 'initial_tr_radius': tr_radius})
    return [scipy_sol, dumper]


# Functions to wrap the constraints to match the scipy definitions
def g1_scipy(x):
    return g1_constraint(x, None, None)


def g2_scipy(x):
    return g2_constraint(x, None, None)


def grad1_scipy(x):
    return g1_gradient(x, None, None).reshape(-1)


def grad2_scipy(x):
    return g2_gradient(x, None, None).reshape(-1)


def hess1_scipy(x, v):
    return float(v) * g1_hessian(x, None, None)


def hess2_scipy(x, v):
    return float(v) * g2_hessian(x, None, None)


class GWrapper:
    """ Wraps the RESSPyLab constraints for use with the scipy solver. """
    def __init__(self, fun, grad, hess, constants):
        """ Constructor.

        :param function fun: Returns the value of the constraint.
        :param function grad: Returns the gradient of the constraint.
        :param function hess: Returns the Hessian of the constraint.
        :param list constants: Any constants used in the constraint functions.
        """
        self.fun = fun
        self.grad = grad
        self.hess = hess
        self.c = constants
        return

    def f(self, x):
        return self.fun(x, self.c, None)

    def gf(self, x):
        return np.reshape(self.grad(x, self.c, None), (1, -1))

    def hf(self, x, v):
        return float(v) * self.hess(x, self.c, None)
