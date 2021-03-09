"""@package vcu_constraints
Constraints for the updated Voce-Chaboche model to maintain a positive tangent modulus and positive parameters.
"""
import numpy as np
from numdifftools import nd_algopy as nda


def g_constraint(x, ep):
    """ Returns the constraint that the tangent modulus is hardening (g function) in standard form.

    :param np.array x: Updated Voce-Chaboche model parameters.
    :param float ep: Plastic strain value.
    :return float: Value of g.
    """
    n_backstresses = int(len(x) - 6) / 2

    g = x[4] * x[5] * np.exp(-x[5] * ep) - x[2] * x[3] * np.exp(-x[3] * ep)
    for i in range(0, n_backstresses):
        ck_ind = 6 + 2 * i
        gk_ind = 7 + 2 * i
        g += -x[ck_ind] * np.exp(-x[gk_ind] * ep)
    return g


def g_gradient(x, ep):
    """ Returns the gradient of the constraint function g.

    :param np.array x: Updated Voce-Chaboche model parameters.
    :param float ep: Plastic strain value.
    :return np.array: (n, 1) Gradient of g, n = len(x).
    """
    n_backstresses = int(len(x) - 6) / 2

    grad = np.zeros((len(x), 1))
    grad[2] = -x[3] * np.exp(-x[3] * ep)
    grad[3] = (-x[2] + x[2] * x[3] * ep) * np.exp(-x[3] * ep)
    grad[4] = x[5] * np.exp(-x[5] * ep)
    grad[5] = (x[4] - x[4] * x[5] * ep) * np.exp(-x[5] * ep)
    for i in range(0, n_backstresses):
        ck_ind = 6 + 2 * i
        gk_ind = 7 + 2 * i
        grad[ck_ind] = -np.exp(-x[gk_ind] * ep)
        grad[gk_ind] = x[ck_ind] * ep * np.exp(-x[gk_ind] * ep)
    return grad


def g_hessian(x, ep):
    """ Returns the Hessian matrix of the constraint g.

    :param np.array x: Updated Voce-Chaboche model parameters.
    :param float ep: Plastic strain value.
    :return np.array: (n, n) Hessian of g, n = len(x).
    """
    n_backstresses = int(len(x) - 6) / 2
    hess = np.zeros((len(x), len(x)))

    # row 2
    hess[2, 3] = (-1. + x[3] * ep) * np.exp(-x[3] * ep)
    # row 3
    hess[3, 2] = (-1. + x[3] * ep) * np.exp(-x[3] * ep)
    hess[3, 3] = (2. * x[2] * ep - x[2] * x[3] * ep ** 2) * np.exp(-x[3] * ep)
    # row 4
    hess[4, 5] = (1. - x[5] * ep) * np.exp(-x[5] * ep)
    # row 5
    hess[5, 4] = (1. - x[5] * ep) * np.exp(-x[5] * ep)
    hess[5, 5] = (-2. * x[4] * ep + x[4] * x[5] * ep ** 2) * np.exp(-x[5] * ep)
    for i in range(0, n_backstresses):
        ck_ind = 6 + 2 * i
        gk_ind = 7 + 2 * i
        # row 6 + 2k
        hess[ck_ind, gk_ind] = ep * np.exp(-x[gk_ind] * ep)
        # row 7  + 2k
        hess[gk_ind, ck_ind] = ep * np.exp(-x[gk_ind] * ep)
        hess[gk_ind, gk_ind] = x[ck_ind] * ep ** 2 * np.exp(-x[gk_ind] * ep)
    return hess


def g1_constraint(x, constants, variables):
    """ Constraint that the initial value of tangent modulus > 0 at ep=0.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    g2 = g_constraint(x, 0.)
    return g2


def g1_gradient(x, constants, variables):
    """ Gradient of constraint that the initial value of tangent modulus > 0 at ep=0.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    grad_g2 = g_gradient(x, 0.)
    return grad_g2


def g1_hessian(x, constants, variables):
    """ Hessian of constraint that the initial value of tangent modulus > 0 at ep=0.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    hess_g2 = g_hessian(x, 0.)
    return hess_g2


def g2_constraint(x, constants, variables):
    """ Constraint for a positive derivative of tangent modulus at ep=0.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    n_backstresses = int((len(x) - 6) / 2)
    g = x[3] ** 2 * x[2] - x[5] ** 2 * x[4]
    for i in range(0, n_backstresses):
        gk_ind = 7 + 2 * i
        ck_ind = 6 + 2 * i
        g += x[ck_ind] * x[gk_ind]
    return g


def g2_gradient(x, constants, variables):
    """ Gradient of constraint for a positive derivative of tangent modulus at ep=0.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    n_backstresses = int(len(x) - 6) / 2

    grad = np.zeros((len(x), 1))
    grad[2] = x[3] ** 2
    grad[3] = 2. * x[2] * x[3]
    grad[4] = -x[5] ** 2
    grad[5] = -2. * x[4] * x[5]
    for i in range(0, n_backstresses):
        ck_ind = 6 + 2 * i
        gk_ind = 7 + 2 * i
        grad[ck_ind] = x[gk_ind]
        grad[gk_ind] = x[ck_ind]
    return grad


def g2_hessian(x, constants, variables):
    """ Hessian of constraint for a positive derivative of tangent modulus at ep=0.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    n_backstresses = int(len(x) - 6) / 2

    hess = np.zeros((len(x), len(x)))
    # 2nd row
    hess[2, 3] = 2. * x[3]
    # 3rd row
    hess[3, 2] = 2. * x[3]
    hess[3, 3] = 2. * x[2]
    # 4th row
    hess[4, 5] = -2. * x[5]
    # 5th row
    hess[5, 4] = -2. * x[5]
    hess[5, 5] = -2. * x[4]
    # cKth and gKth rows
    for i in range(0, n_backstresses):
        ck_ind = 6 + 2 * i
        gk_ind = 7 + 2 * i
        hess[ck_ind, gk_ind] = 1.
        hess[gk_ind, ck_ind] = 1.
    return hess


def positive_x_constraint(x, constants, variables):
    """ Returns the constraints that specify the model parameters are positive.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.

    Notes:
        - The constants dictionary must contain the entry 'min_x': min_val, where min_val >= 0 is the minimum value that
        any x[i] should be able to take (e.g., min_val = 0 specifies that all x[i] should be positive).
    """
    min_val = constants['min_x']
    g = [-xi + min_val for xi in x]
    return g


def positive_x_gradient(x, constants, variables):
    """ Returns gradients of the positive x constraints.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    dg = []
    for i in range(0, len(x)):
        v = np.zeros((len(x), 1))
        v[i] = -1.0
        dg.append(v)

    return dg


def positive_x_hessian(x, constants, variables):
    """ Returns the Hessians of the positive x constraints.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    hg = []
    for i in range(0, len(x)):
        m = np.zeros((len(x), len(x)))
        hg.append(m)

    return hg
