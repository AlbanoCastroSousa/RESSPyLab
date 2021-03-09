"""@package vc_updated
Functions to implement the updated Voce-Chaboche material model and measure its error.
"""
import numpy as np
import pandas as pd
from numdifftools import nd_algopy as nda


def uvc_return_mapping(x_sol, data, tol=1.0e-8, maximum_iterations=1000):
    """ Implements the time integration of the updated Voce-Chaboche material model.

    :param np.array x_sol: Updated Voce-Chaboche model parameters.
    :param pd.DataFrame data: stress-strain data.
    :param float tol: Local Newton tolerance.
    :param int maximum_iterations: maximum iterations in local Newton procedure, raises RuntimeError if exceeded.
    :return dict: History of: stress ('stress'), strain ('strain'), the total error ('error') calculated by the
        updated Voce-Chaboche model, number of iterations for convergence at each increment ('num_its').
    """

    if len(x_sol) < 8:
        raise RuntimeError("No backstresses or using original V-C params.")
    n_param_per_back = 2
    n_basic_param = 6

    # Get material properties
    E = x_sol[0] * 1.0
    sy_0 = x_sol[1] * 1.0
    Q = x_sol[2] * 1.0
    b = x_sol[3] * 1.0
    D = x_sol[4] * 1.0
    a = x_sol[5] * 1.0

    # Set up backstresses
    n_backstresses = int((len(x_sol) - n_basic_param) / n_param_per_back)
    c_k = []
    gamma_k = []
    for i in range(0, n_backstresses):
        c_k.append(x_sol[n_basic_param + n_param_per_back * i])
        gamma_k.append(x_sol[n_basic_param + 1 + n_param_per_back * i])

    # Initialize parameters
    alpha_components = np.zeros(n_backstresses, dtype=object)  # backstress components
    strain = 0.
    stress = 0.
    ep_eq = 0.  # equivalent plastic strain

    error = 0.  # error measure
    sum_abs_de = 0.  # total strain
    stress_sim = 0.0
    stress_test = 0.0
    area_test = 0.0

    stress_track = []
    strain_track = []
    strain_inc_track = []
    iteration_track = []

    loading = np.diff(data['e_true'])
    for increment_number, strain_inc in enumerate(loading):
        strain += strain_inc
        alpha = np.sum(alpha_components)
        yield_stress = sy_0 + Q * (1. - np.exp(-b * ep_eq)) - D * (1. - np.exp(-a * ep_eq))

        trial_stress = stress + E * strain_inc
        relative_stress = trial_stress - alpha
        flow_dir = np.sign(relative_stress)

        yield_condition = np.abs(relative_stress) - yield_stress
        if yield_condition > tol:
            is_converged = False
        else:
            is_converged = True

        # For error
        stress_sim_1 = stress_sim * 1.0
        stress_test_1 = stress_test * 1.0

        # Return mapping if plastic loading
        ep_eq_init = ep_eq
        alpha_init = alpha
        consist_param = 0.
        number_of_iterations = 0
        while is_converged is False and number_of_iterations < maximum_iterations:
            number_of_iterations += 1
            # Isotropic hardening and isotropic modulus
            yield_stress = sy_0 + Q * (1. - np.exp(-b * ep_eq)) - D * (1. - np.exp(-a * ep_eq))
            iso_modulus = Q * b * np.exp(-b * ep_eq) - D * a * np.exp(-a * ep_eq)

            # Kinematic hardening and kinematic modulus
            alpha = 0.
            kin_modulus = 0.
            for i in range(0, n_backstresses):
                e_k = np.exp(-gamma_k[i] * (ep_eq - ep_eq_init))
                alpha += flow_dir * c_k[i] / gamma_k[i] + (alpha_components[i] - flow_dir * c_k[i] / gamma_k[i]) * e_k
                kin_modulus += c_k[i] * e_k - flow_dir * gamma_k[i] * e_k * alpha_components[i]
            delta_alpha = alpha - alpha_init

            # Local Newton step
            numerator = np.abs(relative_stress) - (consist_param * E + yield_stress + flow_dir * delta_alpha)
            denominator = -(E + iso_modulus + kin_modulus)
            consist_param = consist_param - numerator / denominator
            ep_eq = ep_eq_init + consist_param

            if np.abs(numerator) < tol:
                is_converged = True

        # Update the variables
        stress = trial_stress - E * flow_dir * consist_param
        for i in range(0, n_backstresses):
            e_k = np.exp(-gamma_k[i] * (ep_eq - ep_eq_init))
            alpha_components[i] = flow_dir * c_k[i] / gamma_k[i] \
                                  + (alpha_components[i] - flow_dir * c_k[i] / gamma_k[i]) * e_k

        stress_track.append(stress)
        strain_track.append(strain)
        strain_inc_track.append(strain_inc)
        iteration_track.append(number_of_iterations)

        # Calculate the error
        stress_sim = stress * 1.0
        stress_test = data['Sigma_true'].iloc[increment_number + 1]

        sum_abs_de += np.abs(strain_inc)
        area_test += np.abs(strain_inc) * ((stress_test) ** 2 + (stress_test_1) ** 2) / 2.
        error += np.abs(strain_inc) * ((stress_sim - stress_test) ** 2 + (stress_sim_1 - stress_test_1) ** 2) / 2.

        if number_of_iterations >= maximum_iterations:
            print ("Increment number = ", increment_number)
            print ("Parameters = ", x_sol)
            print ("Numerator = ", numerator)
            raise RuntimeError('Return mapping did not converge in ' + str(maximum_iterations) + ' iterations.')

    area = area_test / sum_abs_de
    error = error / sum_abs_de
    return {'stress': stress_track, 'strain': strain_track, 'error': error, 'num_its': iteration_track,
            'area': area}


def sim_curve_uvc(x_sol, test_clean):
    """ Returns the stress-strain approximation of the updated Voce-Chaboche material model to a given strain input.

    :param np.array x_sol: Voce-Chaboche model parameters
    :param DataFrame test_clean: stress-strain data
    :return DataFrame: Voce-Chaboche approximation

    The strain column in the DataFrame is labeled "e_true" and the stress column is labeled "Sigma_true".
    """

    model_output = uvc_return_mapping(x_sol, test_clean)
    strain = np.append([0.], model_output['strain'])
    stress = np.append([0.], model_output['stress'])

    sim_curve = pd.DataFrame(np.array([strain, stress]).transpose(), columns=['e_true', 'Sigma_true'])
    return sim_curve


def error_single_test_uvc(x_sol, test_clean):
    """ Returns the relative error between a test and its approximation using the updated Voce-Chaboche material model.

    :param np.array x_sol: Voce-Chaboche model parameters
    :param DataFrame test_clean: stress-strain data
    :return float: relative error

    The strain column in the DataFrame is labeled "e_true" and the stress column is labeled "Sigma_true".
    """

    model_output = uvc_return_mapping(x_sol, test_clean)
    return model_output['error']


def normalized_error_single_test_uvc(x_sol, test_clean):
    """ Returns the error and the total area of a test and its approximation using the updated Voce-Chaboche
    material model.

    :param np.array x_sol: Voce-Chaboche model parameters
    :param DataFrame test_clean: stress-strain data
    :return list: (float) total error, (float) total area

    The strain column in the DataFrame is labeled "e_true" and the stress column is labeled "Sigma_true".
    """

    model_output = uvc_return_mapping(x_sol, test_clean)
    return [model_output['error'], model_output['area']]


def calc_phi_total(x, data):
    """ Returns the sum of the normalized relative error of the updated Voce-Chaboche material model given x.

    :param np.array x: Updated Voce-Chaboche material model parameters.
    :param list data: (pd.DataFrame) Stress-strain history for each test considered.
    :return float: Normalized error value expressed as a percent (raw value * 100).

    The normalized error is defined in de Sousa and Lignos (2017).
    """
    error_total = 0.
    area_total = 0.
    for d in data:
        error, area = normalized_error_single_test_uvc(x, d)
        error_total += error
        area_total += area

    return np.sqrt(error_total / area_total) * 100.


def test_total_area(x, data):
    """ Returns the total squared area underneath all the tests.

    :param np.array x: Updated Voce-Chaboche material model parameters.
    :param list data: (pd.DataFrame) Stress-strain history for each test considered.
    :return float: Total squared area.
    """
    area_total = 0.
    for d in data:
        _, area = normalized_error_single_test_uvc(x, d)
        area_total += area
    return area_total


def uvc_get_hessian(x, data):
    """ Returns the Hessian of the material model error function for a given set of test data evaluated at x.

    :param np.array x: Updated Voce-Chaboche material model parameters.
    :param list data: (pd.DataFrame) Stress-strain history for each test considered.
    :return np.array: Hessian matrix of the error function.
    """

    def f(xi):
        val = 0.
        for d in data:
            val += error_single_test_uvc(xi, d)
        return val

    hess_fun = nda.Hessian(f)
    return hess_fun(x)


def uvc_consistency_metric(x_base, x_sample, data):
    """ Returns the xi_2 consistency metric from de Sousa and Lignos 2019 using the updated Voce-Chaboche model.

    :param np.array x_base: Updated Voce-Chaboche material model parameters from the base case.
    :param np.array x_sample: Updated Voce-Chaboche material model parameters from the sample case.
    :param list data: (pd.DataFrame) Stress-strain history for each test considered.
    :return float: Increase in quadratic approximation from the base to the sample case.
    """
    x_diff = x_sample - x_base
    hess_base = uvc_get_hessian(x_base, data)
    numerator = np.dot(x_diff, hess_base.dot(x_diff))
    denominator = test_total_area(x_base, data)
    return np.sqrt(numerator / denominator)


def uvc_tangent_modulus(x_sol, data, tol=1.0e-8, maximum_iterations=1000):
    """ Returns the tangent modulus at each strain step.

    :param np.array x_sol: Updated Voce-Chaboche model parameters.
    :param pd.DataFrame data: stress-strain data.
    :param float tol: Local Newton tolerance.
    :param int maximum_iterations: maximum iterations in local Newton procedure, raises RuntimeError if exceeded.
    :return np.ndarray: Tangent modulus array.
    """

    if len(x_sol) < 8:
        raise RuntimeError("No backstresses or using original V-C params.")
    n_param_per_back = 2
    n_basic_param = 6

    # Get material properties
    E = x_sol[0] * 1.0
    sy_0 = x_sol[1] * 1.0
    Q = x_sol[2] * 1.0
    b = x_sol[3] * 1.0
    D = x_sol[4] * 1.0
    a = x_sol[5] * 1.0

    # Set up backstresses
    n_backstresses = int((len(x_sol) - n_basic_param) / n_param_per_back)
    c_k = []
    gamma_k = []
    for i in range(0, n_backstresses):
        c_k.append(x_sol[n_basic_param + n_param_per_back * i])
        gamma_k.append(x_sol[n_basic_param + 1 + n_param_per_back * i])

    # Initialize parameters
    alpha_components = np.zeros(n_backstresses, dtype=object)  # backstress components
    strain = 0.
    stress = 0.
    ep_eq = 0.  # equivalent plastic strain

    stress_track = []
    strain_track = []
    strain_inc_track = []
    iteration_track = []
    tangent_track = []

    loading = np.diff(data['e_true'])
    for increment_number, strain_inc in enumerate(loading):
        strain += strain_inc
        alpha = np.sum(alpha_components)
        yield_stress = sy_0 + Q * (1. - np.exp(-b * ep_eq)) - D * (1. - np.exp(-a * ep_eq))

        trial_stress = stress + E * strain_inc
        relative_stress = trial_stress - alpha
        flow_dir = np.sign(relative_stress)

        yield_condition = np.abs(relative_stress) - yield_stress
        if yield_condition > tol:
            is_converged = False
        else:
            is_converged = True

        # Return mapping if plastic loading
        ep_eq_init = ep_eq
        alpha_init = alpha
        consist_param = 0.
        number_of_iterations = 0
        while is_converged is False and number_of_iterations < maximum_iterations:
            number_of_iterations += 1
            # Isotropic hardening and isotropic modulus
            yield_stress = sy_0 + Q * (1. - np.exp(-b * ep_eq)) - D * (1. - np.exp(-a * ep_eq))
            iso_modulus = Q * b * np.exp(-b * ep_eq) - D * a * np.exp(-a * ep_eq)

            # Kinematic hardening and kinematic modulus
            alpha = 0.
            kin_modulus = 0.
            for i in range(0, n_backstresses):
                e_k = np.exp(-gamma_k[i] * (ep_eq - ep_eq_init))
                alpha += flow_dir * c_k[i] / gamma_k[i] + (alpha_components[i] - flow_dir * c_k[i] / gamma_k[i]) * e_k
                kin_modulus += c_k[i] * e_k - flow_dir * gamma_k[i] * e_k * alpha_components[i]
            delta_alpha = alpha - alpha_init

            # Local Newton step
            numerator = np.abs(relative_stress) - (consist_param * E + yield_stress + flow_dir * delta_alpha)
            denominator = -(E + iso_modulus + kin_modulus)
            consist_param = consist_param - numerator / denominator
            ep_eq = ep_eq_init + consist_param

            if np.abs(numerator) < tol:
                is_converged = True

        # Update the variables
        stress = trial_stress - E * flow_dir * consist_param
        for i in range(0, n_backstresses):
            e_k = np.exp(-gamma_k[i] * (ep_eq - ep_eq_init))
            alpha_components[i] = flow_dir * c_k[i] / gamma_k[i] \
                                  + (alpha_components[i] - flow_dir * c_k[i] / gamma_k[i]) * e_k

        stress_track.append(stress)
        strain_track.append(strain)
        strain_inc_track.append(strain_inc)
        iteration_track.append(number_of_iterations)

        # Calculate the tangent modulus
        if number_of_iterations > 0:
            h_prime = 0.
            for i in range(0, n_backstresses):
                h_prime += c_k[i] - flow_dir * gamma_k[i] * alpha_components[i]
            k_prime = Q * b * np.exp(-b * ep_eq) - D * a * np.exp(-a * ep_eq)
            tangent_track.append(E * (k_prime + h_prime) / (E + k_prime + h_prime))
        else:
            # Elastic loading
            tangent_track.append(E)

    return np.append([0.], np.array(tangent_track))
