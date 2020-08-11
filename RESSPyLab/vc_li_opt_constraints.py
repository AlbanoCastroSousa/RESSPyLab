"""@package sqp_linsearch
Constraints on the original Voce-Chaboche model for limited information optimization.
"""
from numdifftools import nd_algopy as nda
import numpy as np


def g3_vco_upper(x, constants, variables):
    """ Constraint on the maximum ratio of stress at saturation to initial yield stress for the original VC model.

    :param np.ndarray x: Parameters of original Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    max_hardening_to_yield = constants['rho_yield_sup']
    n_backstresses = int((len(x) - 4) / 2)
    sy0 = x[1]
    q_inf = x[2]
    sum_ck_gammak = 0.
    for i in range(n_backstresses):
        c_ind = 4 + 2 * i
        gamma_ind = 5 + 2 * i
        sum_ck_gammak += x[c_ind] / x[gamma_ind]
    return (sy0 + q_inf + sum_ck_gammak) / sy0 - max_hardening_to_yield


def g3_vco_lower(x, constants, variables):
    """ Constraint on the minimum ratio of stress at saturation to initial yield stress for the original VC model.

    :param np.ndarray x: Parameters of original Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    min_hardening_to_yield = constants['rho_yield_inf']
    n_backstresses = int((len(x) - 4) / 2)
    sy0 = x[1]
    q_inf = x[2]
    sum_ck_gammak = 0.
    for i in range(n_backstresses):
        c_ind = 4 + 2 * i
        gamma_ind = 5 + 2 * i
        sum_ck_gammak += x[c_ind] / x[gamma_ind]
    return -(sy0 + q_inf + sum_ck_gammak) / sy0 + min_hardening_to_yield


def g4_vco_upper(x, constants, variables):
    """ Constraint on the maximum ratio of isotropic to combined isotropic/kinematic hardening at saturation for the
    original VC model.

    :param np.ndarray x: Parameters of original Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    iso_kin_ratio_max = constants['rho_iso_sup']
    q_inf = x[2]
    n_backstresses = int((len(x) - 4) / 2)
    sum_ck_gammak = 0.
    for i in range(n_backstresses):
        c_ind = 4 + 2 * i
        gamma_ind = 5 + 2 * i
        sum_ck_gammak += x[c_ind] / x[gamma_ind]
    return q_inf / (q_inf + sum_ck_gammak) - iso_kin_ratio_max


def g4_vco_lower(x, constants, variables):
    """ Constraint on the minimum ratio of isotropic to combined isotropic/kinematic hardening at saturation for the
    original VC model.

    :param np.ndarray x: Parameters of original Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    iso_kin_ratio_min = constants['rho_iso_inf']
    q_inf = x[2]
    n_backstresses = int((len(x) - 4) / 2)
    sum_ck_gammak = 0.
    for i in range(n_backstresses):
        c_ind = 4 + 2 * i
        gamma_ind = 5 + 2 * i
        sum_ck_gammak += x[c_ind] / x[gamma_ind]
    return -q_inf / (q_inf + sum_ck_gammak) + iso_kin_ratio_min


def g5_vco_lower(x, constants, variables):
    """ Constraint on the lower bound ratio of gamma_1 to b for the original VC model.

    :param np.ndarray x: Parameters of original Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    b = x[3]
    gamma1 = x[5]
    gamma_b_ratio_min = constants['rho_gamma_inf']
    return -gamma1 / b + gamma_b_ratio_min


def g5_vco_upper(x, constants, variables):
    """ Constraint on the upper bound ratio of gamma_1 to b for the original VC model.

    :param np.ndarray x: Parameters of original Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    b = x[3]
    gamma1 = x[5]
    gamma_b_ratio_max = constants['rho_gamma_sup']
    return gamma1 / b - gamma_b_ratio_max


def g6_vco_lower(x, constants, variables):
    """ Constraint on the lower bound ratio of gamma_1 to gamma_2 for the original VC model.

    gamma_1 is always x[5] and gamma_2 is always x[7].

    :param np.ndarray x: Parameters of original Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    gamma1 = x[5]
    gamma2 = x[7]
    gamma_1_2_ratio_min = constants['rho_gamma_12_inf']
    return -gamma1 / gamma2 + gamma_1_2_ratio_min


def g6_vco_upper(x, constants, variables):
    """ Constraint on the upper bound ratio of gamma_1 to gamma_2 for the original VC model.

    gamma_1 is always x[5] and gamma_2 is always x[7].

    :param np.ndarray x: Parameters of original Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    gamma1 = x[5]
    gamma2 = x[7]
    gamma_1_2_ratio_max = constants['rho_gamma_12_sup']
    return gamma1 / gamma2 - gamma_1_2_ratio_max


def g_kin_ratio_vco_lower(x, constants, variables):
    c1 = x[4]
    gamma1 = x[5]
    c2 = x[6]
    gamma2 = x[7]
    gamma_kin_ratio_min = constants['rho_kin_ratio_inf']
    return -(c1 / gamma1) / (c2 / gamma2) + gamma_kin_ratio_min


def g_kin_ratio_vco_upper(x, constants, variables):
    c1 = x[4]
    gamma1 = x[5]
    c2 = x[6]
    gamma2 = x[7]
    gamma_kin_ratio_max = constants['rho_kin_ratio_sup']
    return (c1 / gamma1) / (c2 / gamma2) - gamma_kin_ratio_max


# Gradients and Hessians of all the above constraints
def g3_vco_lower_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g3_vco_lower(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g3_vco_upper_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g3_vco_upper(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g4_vco_lower_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g4_vco_lower(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g4_vco_upper_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g4_vco_upper(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g5_vco_lower_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g5_vco_lower(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g5_vco_upper_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g5_vco_upper(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g6_vco_lower_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g6_vco_lower(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return grad


def g6_vco_upper_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g6_vco_upper(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return grad


def g_kin_ratio_vco_lower_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g_kin_ratio_vco_lower(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return grad


def g_kin_ratio_vco_upper_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g_kin_ratio_vco_upper(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return grad


# Hessians

def g3_vco_lower_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g3_vco_lower(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g3_vco_upper_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g3_vco_upper(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g4_vco_lower_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g4_vco_lower(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g4_vco_upper_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g4_vco_upper(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g5_vco_lower_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g5_vco_lower(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g5_vco_upper_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g5_vco_upper(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g6_vco_lower_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g6_vco_lower(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g6_vco_upper_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g6_vco_upper(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g_kin_ratio_vco_lower_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g_kin_ratio_vco_lower(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g_kin_ratio_vco_upper_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g_kin_ratio_vco_upper(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess
