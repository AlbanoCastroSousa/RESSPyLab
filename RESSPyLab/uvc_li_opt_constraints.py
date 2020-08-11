"""@package vcu_constraints
Constraints for the updated Voce-Chaboche model for limited information opt.
"""
import numpy as np
from numdifftools import nd_algopy as nda


def g3_upper(x, constants, variables):
    """ Constraint on the maximum ratio of stress at saturation to initial yield stress.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    max_hardening_to_yield = constants['rho_yield_sup']
    n_backstresses = int((len(x) - 6) / 2)
    sy0 = x[1]
    q_inf = x[2]
    d_inf = x[4]
    sum_ck_gammak = 0.
    for i in range(n_backstresses):
        c_ind = 6 + 2 * i
        gamma_ind = 7 + 2 * i
        sum_ck_gammak += x[c_ind] / x[gamma_ind]
    return (sy0 + q_inf - d_inf + sum_ck_gammak) / sy0 - max_hardening_to_yield


def g3_lower(x, constants, variables):
    """ Constraint on the minimum ratio of stress at saturation to initial yield stress.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    min_hardening_to_yield = constants['rho_yield_inf']
    n_backstresses = int((len(x) - 6) / 2)
    sy0 = x[1]
    q_inf = x[2]
    d_inf = x[4]
    sum_ck_gammak = 0.
    for i in range(n_backstresses):
        c_ind = 6 + 2 * i
        gamma_ind = 7 + 2 * i
        sum_ck_gammak += x[c_ind] / x[gamma_ind]
    return -(sy0 + q_inf - d_inf + sum_ck_gammak) / sy0 + min_hardening_to_yield


def g4_upper(x, constants, variables):
    """ Constraint on the maximum ratio of isotropic to combined isotropic/kinematic hardening at saturation.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    iso_kin_ratio_max = constants['rho_iso_sup']
    q_inf = x[2]
    n_backstresses = int((len(x) - 6) / 2)
    sum_ck_gammak = 0.
    for i in range(n_backstresses):
        c_ind = 6 + 2 * i
        gamma_ind = 7 + 2 * i
        sum_ck_gammak += x[c_ind] / x[gamma_ind]
    return q_inf / (q_inf + sum_ck_gammak) - iso_kin_ratio_max


def g4_lower(x, constants, variables):
    """ Constraint on the minimum ratio of isotropic to combined isotropic/kinematic hardening at saturation.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    iso_kin_ratio_min = constants['rho_iso_inf']
    q_inf = x[2]
    n_backstresses = int((len(x) - 6) / 2)
    sum_ck_gammak = 0.
    for i in range(n_backstresses):
        c_ind = 6 + 2 * i
        gamma_ind = 7 + 2 * i
        sum_ck_gammak += x[c_ind] / x[gamma_ind]
    return -q_inf / (q_inf + sum_ck_gammak) + iso_kin_ratio_min


def g5_lower(x, constants, variables):
    """ Constraint on the lower bound ratio of gamma_1 to b for the updated VC model.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    b = x[3]
    gamma1 = x[7]
    gamma_b_ratio_min = constants['rho_gamma_b_inf']
    return -gamma1 / b + gamma_b_ratio_min


def g5_upper(x, constants, variables):
    """ Constraint on the upper bound ratio of gamma_1 to b for the updated VC model.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    b = x[3]
    gamma1 = x[7]
    gamma_b_ratio_max = constants['rho_gamma_b_sup']
    return gamma1 / b - gamma_b_ratio_max


def g6_lower(x, constants, variables):
    """ Constraint on the lower bound ratio of gamma_1 to gamma_2 for the updated VC model.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.

    gamma_1 is always x[7] and gamma_2 is always x[9].
    """
    gamma1 = x[7]
    gamma2 = x[9]
    gamma_1_2_ratio_min = constants['rho_gamma_12_inf']
    return -gamma1 / gamma2 + gamma_1_2_ratio_min


def g6_upper(x, constants, variables):
    """ Constraint on the upper bound ratio of gamma_1 to gamma_2 for the updated VC model.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.

    gamma_1 is always x[7] and gamma_2 is always x[9].
    """
    gamma1 = x[7]
    gamma2 = x[9]
    gamma_1_2_ratio_max = constants['rho_gamma_12_sup']
    return gamma1 / gamma2 - gamma_1_2_ratio_max


def g7_lower(x, constants, variables):
    """ Constraint on the lower bound ratio of D_inf to the total hardening for the updated VC model.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    q_inf = x[2]
    d_inf = x[4]
    n_backstresses = int((len(x) - 6) / 2)
    sum_c_gamma = 0.
    for i in range(n_backstresses):
        sum_c_gamma += x[6 + 2 * i] / x[7 + 2 * i]
    d_ratio_min = constants['rho_d_inf']
    return -d_inf / (q_inf + sum_c_gamma) + d_ratio_min


def g7_upper(x, constants, variables):
    """ Constraint on the upper bound ratio of D_inf to the total hardening for the updated VC model.

    :param np.ndarray x: Parameters of updated Voce-Chaboche model.
    :param dict constants: Defines the constants for the constraint.
    :param dict variables: Defines constraint values that depend on x.
    :return float: Value of the constraint in standard form.
    """
    q_inf = x[2]
    d_inf = x[4]
    n_backstresses = int((len(x) - 6) / 2)
    sum_c_gamma = 0.
    for i in range(n_backstresses):
        sum_c_gamma += x[6 + 2 * i] / x[7 + 2 * i]
    d_ratio_max = constants['rho_d_sup']
    return d_inf / (q_inf + sum_c_gamma) - d_ratio_max


""" Gradients and Hessians of the above constraints. """


def g3_lower_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g3_lower(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g3_upper_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g3_upper(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g4_lower_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g4_lower(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g4_upper_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g4_upper(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g5_lower_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g5_lower(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g5_upper_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g5_upper(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return np.reshape(grad, (-1, 1))


def g6_lower_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g6_lower(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return grad


def g6_upper_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g6_upper(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return grad


def g7_lower_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g7_lower(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return grad


def g7_upper_gradient(x, constants, variables):
    fun_wrapper = lambda x1: g7_upper(x1, constants, variables)
    grad_fun = nda.Gradient(fun_wrapper)
    grad = grad_fun(x)
    return grad


def g3_lower_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g3_lower(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g3_upper_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g3_upper(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g4_lower_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g4_lower(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g4_upper_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g4_upper(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g5_lower_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g5_lower(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g5_upper_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g5_upper(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g6_lower_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g6_lower(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g6_upper_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g6_upper(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g7_lower_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g7_lower(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess


def g7_upper_hessian(x, constants, variables):
    fun_wrapper = lambda x1: g7_upper(x1, constants, variables)
    hess_fun = nda.Hessian(fun_wrapper)
    hess = hess_fun(x)
    return hess
