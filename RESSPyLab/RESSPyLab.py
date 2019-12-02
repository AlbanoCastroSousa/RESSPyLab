# -*- coding: utf-8 -*-
"""@package RESSPyLab
Material model and solvers for the original Voce-Chaboche model.
"""
import numpy as np
import pandas as pd
import codecs
import numdifftools.nd_algopy as nda


def VCsimCurve(x_sol, testClean):
    sy_0 = x_sol[1] * 1.0
    E = x_sol[0] * 1.0
    Q = x_sol[2] * 1.0
    b = x_sol[3] * 1.0

    nBack = int((len(x_sol) - 4) / 2)

    chabCoef = []

    for i in range(nBack):
        chabCoef.append([x_sol[4 + (2 * i)] * 1.0, x_sol[5 + (2 * i)] * 1.0])

    sigma = 0.0
    ep_eq = 0.0
    e_p = 0.0
    e_el = 0.0
    e = 0.0

    alphaN = np.zeros(nBack, dtype=object) * 1.0
    sy = sy_0 * 1.0

    Tol = 1e-10

    e_true_calc = [0]
    sigma_calc = [0]
    ep_eq_calc = [0]

    lineCounter = 0

    Phi_test = 0.0

    sum_abs_de = 0.0

    sigma_SimN = 0.0
    sigma_testN = 0.0

    testClean['delta_e'] = testClean['e_true'].shift(-1) - testClean['e_true']
    loading = testClean['delta_e'].dropna().values

    for de in loading:

        lineCounter = lineCounter + 1

        e = e + de
        # e_el=e_el+de

        alpha = np.sum(alphaN)

        sigma = sigma + E * de

        phi = (sigma - alpha) ** 2 - sy ** 2

        ep_eq_n_1 = ep_eq * 1.0

        alphaN_1 = alphaN * 1.0

        sy_N_1 = sy * 1.0

        sigma_SimN_1 = sigma_SimN * 1.0
        sigma_testN_1 = sigma_testN * 1.0

        if phi > Tol:

            nitMax = 1000

            for nit in np.arange(nitMax):

                aux = E * 1.0
                for k in np.arange(nBack):
                    aux = aux + np.sign(sigma - alpha) * chabCoef[k][0] - chabCoef[k][1] * alphaN[k]

                dit = (-2. * alpha + 2. * sigma) * aux + 2. * sy * Q * b * np.exp(-b * ep_eq)

                dep = (phi / (dit))

                scale = 1.0

                # scale to ensure that the Newton step does not overshoot

                if abs(dep) > abs(sigma / E):
                    dep = np.sign(dep) * 0.95 * abs(sigma / E)
                    scale = 0.95 * abs(sigma / E) / abs(dep)

                ## Update variables ##

                ep_eq = ep_eq + np.abs(dep)

                e_p = e_p + dep

                sigma = sigma - E * dep

                sy = sy_0 + Q * (1. - np.exp(-b * ep_eq))

                for k in np.arange(nBack):
                    c_k = chabCoef[k][0]
                    gam_k = chabCoef[k][1]
                    alphaN[k] = np.sign(sigma - alpha) * c_k / gam_k - (
                            np.sign(sigma - alpha) * c_k / gam_k - alphaN_1[k]) * np.exp(
                        -gam_k * (ep_eq - ep_eq_n_1))

                alpha = np.sum(alphaN)

                phi = (sigma - alpha) ** 2 - sy ** 2

                if abs(phi) < Tol:
                    break

                # if nit-2==nitMax:
                # print ('Warning convergence not reached in nonlinear loop!!!')

        sigma_SimN = sigma * 1.0
        sigma_testN = testClean['Sigma_true'].iloc[lineCounter]

        e_true_calc.append(e)
        sigma_calc.append(sigma)
        ep_eq_calc.append(ep_eq)

        # difSigN=sigma-testClean

        sum_abs_de = sum_abs_de + np.abs(de)

        # Square of the area under the increment with the trapezoidal rule

        Phi_test = Phi_test + np.abs(de) * ((sigma_SimN - sigma_testN) ** 2 + (sigma_SimN_1 - sigma_testN_1) ** 2) / 2.

    Phi_test = Phi_test / sum_abs_de

    simCurve = pd.DataFrame(np.array([e_true_calc, sigma_calc]).transpose(), columns=['e_true', 'Sigma_true'])

    return simCurve


def errorTest_scl(x_sol, testClean):
    sy_0 = x_sol[1] * 1.0
    E = x_sol[0] * 1.0
    Q = x_sol[2] * 1.0
    b = x_sol[3] * 1.0

    nBack = int((len(x_sol) - 4) / 2)

    chabCoef = []

    for i in range(nBack):
        chabCoef.append([x_sol[4 + (2 * i)] * 1.0, x_sol[5 + (2 * i)] * 1.0])

    sigma = 0.0
    ep_eq = 0.0
    e_p = 0.0
    e_el = 0.0
    e = 0.0

    alphaN = np.zeros(nBack, dtype=object) * 1.0
    sy = sy_0 * 1.0

    Tol = 1e-10

    e_true_calc = [0]
    sigma_calc = [0]
    ep_eq_calc = [0]

    lineCounter = 0

    Phi_test = 0.0

    sum_abs_de = 0.0

    sigma_SimN = 0.0
    sigma_testN = 0.0

    testClean['delta_e'] = testClean['e_true'].shift(-1) - testClean['e_true']
    loading = testClean['delta_e'].dropna().values

    for de in loading:

        lineCounter = lineCounter + 1

        e = e + de
        # e_el=e_el+de

        alpha = np.sum(alphaN)

        sigma = sigma + E * de

        phi = (sigma - alpha) ** 2 - sy ** 2

        ep_eq_n_1 = ep_eq * 1.0

        alphaN_1 = alphaN * 1.0

        sy_N_1 = sy * 1.0

        sigma_SimN_1 = sigma_SimN * 1.0
        sigma_testN_1 = sigma_testN * 1.0

        if phi > Tol:

            nitMax = 1000

            for nit in np.arange(nitMax):

                aux = E * 1.0
                for k in np.arange(nBack):
                    aux = aux + np.sign(sigma - alpha) * chabCoef[k][0] - chabCoef[k][1] * alphaN[k]

                dit = (-2. * alpha + 2. * sigma) * aux + 2. * sy * Q * b * np.exp(-b * ep_eq)

                dep = (phi / (dit))

                scale = 1.0

                # scale to ensure that the Newton step does not overshoot

                if abs(dep) > abs(sigma / E):
                    dep = np.sign(dep) * 0.95 * abs(sigma / E)
                    scale = 0.95 * abs(sigma / E) / abs(dep)

                ## Update variables ##

                ep_eq = ep_eq + np.abs(dep)

                e_p = e_p + dep

                sigma = sigma - E * dep

                sy = sy_0 + Q * (1. - np.exp(-b * ep_eq))

                for k in np.arange(nBack):
                    c_k = chabCoef[k][0]
                    gam_k = chabCoef[k][1]
                    alphaN[k] = np.sign(sigma - alpha) * c_k / gam_k - (
                            np.sign(sigma - alpha) * c_k / gam_k - alphaN_1[k]) * np.exp(
                        -gam_k * (ep_eq - ep_eq_n_1))

                alpha = np.sum(alphaN)

                phi = (sigma - alpha) ** 2 - sy ** 2

                if abs(phi) < Tol:
                    break

                # if nit-2==nitMax:
                # print ('Warning convergence not reached in nonlinear loop!!!')

        sigma_SimN = sigma * 1.0
        sigma_testN = testClean['Sigma_true'].iloc[lineCounter]

        e_true_calc.append(e)
        sigma_calc.append(sigma)
        ep_eq_calc.append(ep_eq)

        # difSigN=sigma-testClean

        sum_abs_de = sum_abs_de + np.abs(de)

        # Square of the area under the increment with the trapezoidal rule

        Phi_test = Phi_test + np.abs(de) * ((sigma_SimN - sigma_testN) ** 2 + (sigma_SimN_1 - sigma_testN_1) ** 2) / 2.

    Phi_test = Phi_test / sum_abs_de

    return Phi_test


def errorEnsemble_nda(x_sol):
    Phi_ensemble = 0.0

    for cleanedTest in arrayCleanTests:
        Phi_ensemble = Phi_ensemble + errorTest_scl(x_sol, cleanedTest)

    # barrier for function for zero constraints
    for barrier in range(len(x_sol)):
        # if x_sol > 0:
        Phi_ensemble = Phi_ensemble + 1. / x_sol[barrier] ** 2

    return Phi_ensemble


##### Steihaug-Toint truncated conjugated gradient method #####

def steihaug(Q, b, Delta):
    x_1 = np.zeros(len(b))
    d = -b * 1.0

    flag = 0

    for i in range(len(b) + 1):

        x_prev = x_1

        if np.dot(d, np.dot(Q, d)) < 0:
            flag = 1

            a_ = np.dot(d, d)
            b_ = np.dot(2 * x_1, d)
            c_ = np.dot(x_1, x_1) - Delta ** 2

            lambda_ = (-b_ + np.sqrt(b_ ** 2 - 4 * a_ * c_)) / (2 * a_)

            x_stei = x_1 + lambda_ * d

            break

        alpha = -np.dot(d, np.dot(Q, x_1) + b) / np.dot(d, np.dot(Q, d))

        x_1 = x_prev + alpha * d

        if np.sqrt(np.dot(x_1, x_1)) > Delta:
            flag = 1

            a_ = np.dot(d, d)
            b_ = np.dot(2 * x_prev, d)
            c_ = np.dot(x_prev, x_prev) - Delta ** 2

            lambda_ = (-b_ + np.sqrt(b_ ** 2 - 4 * a_ * c_)) / (2 * a_)

            x_stei = x_prev + lambda_ * d

            break

        beta = np.dot(np.dot(Q, x_1) + b, np.dot(Q, x_1) + b) / np.dot(np.dot(Q, x_prev) + b, np.dot(Q, x_prev) + b)

        d = -np.dot(Q, x_1) - b + beta * d

        if np.sqrt(np.dot(np.dot(Q, x_1) + b, np.dot(Q, x_1) + b)) < 1e-10:
            break

    if flag == 0:
        x_stei = x_1

    return x_stei


def NTR_SVD_Solver(f, df, Hf, x):
    ##### Newton's method with trust region with Steihaug-Toint truncated conjugated gradient method with SVD preconditioning ######

    # Initialization

    dk = np.zeros(len(x))

    Delta = 10
    Tol = 1e-10

    eta1 = 0.01
    eta2 = 0.9

    nitNMTRmax = int(1e6)

    gradPhi_fun = df
    HessPhi_fun = Hf

    gradPhi_test = gradPhi_fun(x)
    HessPhi_test = HessPhi_fun(x)

    for nit in range(nitNMTRmax):

        Phi_test_k = f(x) * 1.0

        # Solve trust-region sub-problem with steihaug-toint method and a preconditioned hessian

        U, D, V = np.linalg.svd(HessPhi_test)

        # Filtering out ill-conditioned space

        D_S = []
        D_S_inv = []

        for sing in D:
            if sing < 1e-15 * D.max():
                D_S.append(0.)
                D_S_inv.append(0)
            else:
                D_S.append(sing ** -0.5)
                D_S_inv.append(sing ** 0.5)

        D_S = np.array(D_S)
        D_S_inv = np.array(D_S_inv)

        S = np.dot(np.dot(V.transpose(), np.diag(D_S)), U.transpose())
        S_inv = np.dot(np.dot(U, np.diag(D_S_inv)), V)

        HessPhi_test_S = np.dot(np.dot(S.transpose(), HessPhi_test), S)

        gradPhi_test_S = np.dot(S.transpose(), gradPhi_test)

        dk = steihaug(HessPhi_test_S, gradPhi_test_S, Delta)

        # Bring back the scaled step

        dk = np.dot(S, dk)

        # Prevent the step from over-shooting the barrier

        x_trial = x + dk

        barrier_scale = 1.0

        if np.min(x_trial) < Tol:
            ind = np.argmin(x_trial)
            barrier_scale = (Tol - x[ind]) / dk[ind]

        dk = barrier_scale * dk

        model_k = Phi_test_k * 1.0

        model_k1 = Phi_test_k + np.dot(dk, gradPhi_test) + 0.5*np.dot(dk, np.dot(HessPhi_test, dk))

        Phi_test_k1 = f(x + dk) * 1.0

        rho = (Phi_test_k - Phi_test_k1) / (model_k - model_k1)

        if ((model_k - model_k1) < 1e-14 and (Phi_test_k - Phi_test_k1) > 0) or np.abs(
                (Phi_test_k - Phi_test_k1) / Phi_test_k) < 1e-14:

            x = x + dk

            gradPhi_test = gradPhi_fun(x)
            HessPhi_test = HessPhi_fun(x)

            if rho >= 0.9:
                Delta = 2. * Delta


        elif rho < eta1 or (Phi_test_k - Phi_test_k1) < 0:

            Delta = 0.5 * np.sqrt(np.dot(np.dot(S_inv, dk), np.dot(S_inv, dk)))

        else:

            x = x + dk

            gradPhi_test = gradPhi_fun(x)
            HessPhi_test = HessPhi_fun(x)

            if rho >= 0.9:
                Delta = 2. * Delta

        norm_grad = np.sqrt(np.dot(gradPhi_test, gradPhi_test))

        print ('It. ' + str(nit) + ' ; Function: ' + str(Phi_test_k) + ' ; norm_grad: ' + str(norm_grad))

        if norm_grad < Tol:
            break

    return x


def NTR_J_Solver(f, df, Hf, x):
    ##### Newton's method with trust region with Steihaug-Toint truncated conjugated gradient method with Jacobi preconditioning ######

    # Initialization

    dk = np.zeros(len(x))

    Delta = 10
    Tol = 1e-10

    eta1 = 0.01
    eta2 = 0.9

    nitNMTRmax = int(1e6)

    gradPhi_fun = df
    HessPhi_fun = Hf

    gradPhi_test = gradPhi_fun(x)
    HessPhi_test = HessPhi_fun(x)
    reboot = 0

    for nit in range(nitNMTRmax):

        Phi_test_k = f(x) * 1.0

        # Solve trust-region sub-problem with steihaug-toint method and a scaled hessian

        Diag = np.power(np.abs(np.diag(HessPhi_test)), 0.5)
        Diag[np.abs(Diag) < 1e-10 * np.max(Diag)] = 1
        S = np.diag(1 / Diag)

        # S=np.diag(1/np.power(np.abs(np.diag(HessPhi_test)),0.5))

        HessPhi_test_S = np.dot(np.dot(S.transpose(), HessPhi_test), S)

        gradPhi_test_S = np.dot(S.transpose(), gradPhi_test)

        dk = steihaug(HessPhi_test_S, gradPhi_test_S, Delta)

        # Bring back the scaled step

        dk = np.dot(S, dk)

        # Prevent the step from over-shooting the barrier

        x_trial = x + dk

        barrier_scale = 1.0

        if np.min(x_trial) < Tol:
            ind = np.argmin(x_trial)
            barrier_scale = (Tol - x[ind]) / dk[ind]

        dk = barrier_scale * dk

        model_k = Phi_test_k * 1.0

        model_k1 = Phi_test_k + np.dot(dk, gradPhi_test) + 0.5*np.dot(dk, np.dot(HessPhi_test, dk))

        Phi_test_k1 = f(x + dk) * 1.0

        rho = (Phi_test_k - Phi_test_k1) / (model_k - model_k1)

        if ((model_k - model_k1) < 1e-14 and (Phi_test_k - Phi_test_k1) > 0) or np.abs(
                (Phi_test_k - Phi_test_k1) / Phi_test_k) < 1e-14:
            x = x + dk

            gradPhi_test = gradPhi_fun(x)
            HessPhi_test = HessPhi_fun(x)

            if rho >= 0.9:
                Delta = 2. * Delta


        elif rho < eta1 or (Phi_test_k - Phi_test_k1) < 0:
            S_inv = np.diag(np.power(np.abs(np.diag(HessPhi_test)), 0.5))
            Delta = 0.5 * np.sqrt(np.dot(np.dot(S_inv, dk), np.dot(S_inv, dk)))

        else:
            x = x + dk

            gradPhi_test = gradPhi_fun(x)
            HessPhi_test = HessPhi_fun(x)

            if rho >= 0.9:
                Delta = 2. * Delta

        norm_grad = np.sqrt(np.dot(gradPhi_test, gradPhi_test))

        print ('It. ' + str(nit) + ' ; Function: ' + str(Phi_test_k) + ' ; norm_grad: ' + str(norm_grad))

        if norm_grad < Tol:
            break

    return x


def VCopt_SVD(x_0, listCleanTests):
    global arrayCleanTests

    arrayCleanTests = list(listCleanTests)

    grad_error_fun = nda.Gradient(errorEnsemble_nda)

    Hess_error_fun = nda.Hessian(errorEnsemble_nda)

    x_sol = NTR_SVD_Solver(errorEnsemble_nda, grad_error_fun, Hess_error_fun, x_0)

    return x_sol


def VCopt_J(x_0, listCleanTests):
    global arrayCleanTests

    arrayCleanTests = list(listCleanTests)

    grad_error_fun = nda.Gradient(errorEnsemble_nda)

    Hess_error_fun = nda.Hessian(errorEnsemble_nda)

    x_sol = NTR_J_Solver(errorEnsemble_nda, grad_error_fun, Hess_error_fun, x_0)

    return x_sol


############################################################
############################################################
############################################################
#########                                   ################
#########   Augmented Lagrangian approach   ################
#########                                   ################
############################################################
############################################################
############################################################


def Lagrangian(Lag_arg):  # This function defines the Lagrangian for inequality contraints

    x_sol, Delta, miu, c, g, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup = Lag_arg

    Phi_ensemble = 0.0

    for cleanedTest in arrayCleanTests:
        Phi_ensemble = Phi_ensemble + errorTest_scl(x_sol, cleanedTest)

    Lagrangian = 0.0

    Lagrangian = Lagrangian + Phi_ensemble

    # Inequality constraints

    for j, miu_j in enumerate(miu):
        Lagrangian = Lagrangian + 1 / (2. * c) * (max(0., miu_j + c * g[j]) ** 2 - miu_j ** 2)

    return Lagrangian


def gradLag(Lag_arg,
            grad_error_fun):  # this function assembles the gradient of the Lagrangian for the isotropic rho_inf and rho_ sup conditions

    x_sol, Delta, miu, c, g, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup = Lag_arg

    nBack = int((len(x_sol) - 4) / 2)

    gradLag = np.zeros(len(x_sol))

    gradLag = gradLag + grad_error_fun(x_sol)

    for j, miu_j in enumerate(miu):

        if max(0., miu_j + c * g[j]) > 0.:

            if j in range(len(x_sol)):
                grad_g_j = np.zeros(len(x_sol))
                grad_g_j[j] = -1.0

                gradLag = gradLag + (c * g[j] + miu_j) * grad_g_j

                # print ("grad_g_j for max>0 in x:"+str((c*g[j]+miu_j)*grad_g_j))


            elif j == len(x_sol):  # r = n+1 rho_iso_inf

                grad_g_j = np.zeros(len(x_sol))
                grad_g_j[2] = rho_iso_inf - 1

                for k in range(nBack):
                    grad_g_j[4 + 2 * k] = rho_iso_inf / x_sol[5 + 2 * k]
                    grad_g_j[5 + 2 * k] = -rho_iso_inf * x_sol[4 + 2 * k] / x_sol[5 + 2 * k] ** 2

                gradLag = gradLag + (c * g[j] + miu_j) * grad_g_j


            elif j == len(x_sol) + 1:  # r = n+2 rho_iso_sup

                grad_g_j = np.zeros(len(x_sol))
                grad_g_j[2] = 1 - rho_iso_sup

                for k in range(nBack):
                    grad_g_j[4 + 2 * k] = -rho_iso_sup / x_sol[5 + 2 * k]
                    grad_g_j[5 + 2 * k] = rho_iso_sup * x_sol[4 + 2 * k] / x_sol[5 + 2 * k] ** 2

                gradLag = gradLag + (c * g[j] + miu_j) * grad_g_j


            elif j == len(x_sol) + 2:  # r = n+2 rho_yield_inf

                grad_g_j = np.zeros(len(x_sol))
                grad_g_j[1] = rho_yield_inf - 1
                grad_g_j[2] = -1

                for k in range(nBack):
                    grad_g_j[4 + 2 * k] = -1 / x_sol[5 + 2 * k]
                    grad_g_j[5 + 2 * k] = x_sol[4 + 2 * k] / x_sol[5 + 2 * k] ** 2


            elif j == len(x_sol) + 3:  # r = n+2 rho_yield_sup

                grad_g_j = np.zeros(len(x_sol))
                grad_g_j[1] = 1 - rho_yield_sup
                grad_g_j[2] = 1

                for k in range(nBack):
                    grad_g_j[4 + 2 * k] = 1 / x_sol[5 + 2 * k]
                    grad_g_j[5 + 2 * k] = -x_sol[4 + 2 * k] / x_sol[5 + 2 * k] ** 2

                gradLag = gradLag + (c * g[j] + miu_j) * grad_g_j

    return gradLag


def hessLag(Lag_arg,
            Hess_error_fun,
            grad_error_fun):  # this function assembles the Hessian of the Lagrangian for the isotropic rho_in and rho_ sup conditions

    x_sol, Delta, miu, c, g, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup = Lag_arg

    nBack = int((len(x_sol) - 4) / 2)

    Hess_Lag = np.zeros((len(x_sol), len(x_sol)))

    Hess_Lag = Hess_Lag + Hess_error_fun(x_sol)

    gradLag = np.zeros(len(x_sol))

    gradLag = gradLag + grad_error_fun(x_sol)

    for j, miu_j in enumerate(miu):

        if max(0., miu_j + c * g[j]) > 0.:

            if j in range(len(x_sol)):

                grad_g_j = np.zeros(len(x_sol))
                grad_g_j[j] = -1.0

                Hess_g_j = np.zeros((len(x_sol), len(x_sol)))

                Hess_Lag = Hess_Lag + (c * g[j] + miu_j) * Hess_g_j + c * np.dot(np.array([grad_g_j]).transpose(),
                                                                                 np.array([grad_g_j]))


            elif j == len(x_sol):  # r = n+1 rho_inf

                grad_g_j = np.zeros(len(x_sol))
                grad_g_j[2] = rho_iso_inf - 1

                for k in range(nBack):
                    grad_g_j[4 + 2 * k] = rho_iso_inf / x_sol[5 + 2 * k]
                    grad_g_j[5 + 2 * k] = -rho_iso_inf * x_sol[4 + 2 * k] / x_sol[5 + 2 * k] ** 2

                Hess_g_j = np.zeros((len(x_sol), len(x_sol)))

                for k in range(nBack):
                    Hess_g_j[4 + 2 * k, 5 + 2 * k] = -rho_iso_inf / x_sol[5 + 2 * k] ** 2
                    Hess_g_j[5 + 2 * k, 4 + 2 * k] = -rho_iso_inf / x_sol[5 + 2 * k] ** 2
                    Hess_g_j[5 + 2 * k, 5 + 2 * k] = rho_iso_inf * x_sol[4 + 2 * k] / x_sol[5 + 2 * k] ** 3

                Hess_Lag = Hess_Lag + (c * g[j] + miu_j) * Hess_g_j + c * np.dot(np.array([grad_g_j]).transpose(),
                                                                                 np.array([grad_g_j]))

            elif j == len(x_sol) + 1:  # r = n+2 rho_sup

                grad_g_j = np.zeros(len(x_sol))
                grad_g_j[2] = 1 - rho_iso_sup

                for k in range(nBack):
                    grad_g_j[4 + 2 * k] = -rho_iso_sup / x_sol[5 + 2 * k]
                    grad_g_j[5 + 2 * k] = rho_iso_sup * x_sol[4 + 2 * k] / x_sol[5 + 2 * k] ** 2

                Hess_g_j = np.zeros((len(x_sol), len(x_sol)))

                for k in range(nBack):
                    Hess_g_j[4 + 2 * k, 5 + 2 * k] = rho_iso_sup / x_sol[5 + 2 * k] ** 2
                    Hess_g_j[5 + 2 * k, 4 + 2 * k] = rho_iso_sup / x_sol[5 + 2 * k] ** 2
                    Hess_g_j[5 + 2 * k, 5 + 2 * k] = -rho_iso_sup * x_sol[4 + 2 * k] / x_sol[5 + 2 * k] ** 3

                Hess_Lag = Hess_Lag + (c * g[j] + miu_j) * Hess_g_j + c * np.dot(np.array([grad_g_j]).transpose(),
                                                                                 np.array([grad_g_j]))


            elif j == len(x_sol) + 2:  # r = n+2 rho_yield_inf

                grad_g_j = np.zeros(len(x_sol))
                grad_g_j[1] = rho_yield_inf - 1
                grad_g_j[2] = -1

                for k in range(nBack):
                    grad_g_j[4 + 2 * k] = -1 / x_sol[5 + 2 * k]
                    grad_g_j[5 + 2 * k] = x_sol[4 + 2 * k] / x_sol[5 + 2 * k] ** 2

                Hess_g_j = np.zeros((len(x_sol), len(x_sol)))

                for k in range(nBack):
                    Hess_g_j[4 + 2 * k, 5 + 2 * k] = 1 / x_sol[5 + 2 * k] ** 2
                    Hess_g_j[5 + 2 * k, 4 + 2 * k] = 1 / x_sol[5 + 2 * k] ** 2
                    Hess_g_j[5 + 2 * k, 5 + 2 * k] = -x_sol[4 + 2 * k] / x_sol[5 + 2 * k] ** 3

                Hess_Lag = Hess_Lag + (c * g[j] + miu_j) * Hess_g_j + c * np.dot(np.array([grad_g_j]).transpose(),
                                                                                 np.array([grad_g_j]))

            elif j == len(x_sol) + 3:  # r = n+2 rho_yield_sup

                grad_g_j = np.zeros(len(x_sol))
                grad_g_j[1] = 1 - rho_yield_sup
                grad_g_j[2] = 1

                for k in range(nBack):
                    grad_g_j[4 + 2 * k] = 1 / x_sol[5 + 2 * k]
                    grad_g_j[5 + 2 * k] = -x_sol[4 + 2 * k] / x_sol[5 + 2 * k] ** 2

                Hess_g_j = np.zeros((len(x_sol), len(x_sol)))

                for k in range(nBack):
                    Hess_g_j[4 + 2 * k, 5 + 2 * k] = -1. / x_sol[5 + 2 * k] ** 2
                    Hess_g_j[5 + 2 * k, 4 + 2 * k] = -1. / x_sol[5 + 2 * k] ** 2
                    Hess_g_j[5 + 2 * k, 5 + 2 * k] = x_sol[4 + 2 * k] / x_sol[5 + 2 * k] ** 3

                Hess_Lag = Hess_Lag + (c * g[j] + miu_j) * Hess_g_j + c * np.dot(np.array([grad_g_j]).transpose(),
                                                                                 np.array([grad_g_j]))

    return Hess_Lag


def makeG(x, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup):
    sum_of_ck_gam_k = 0.0

    nBack = int((len(x) - 4) / 2)

    for k in range(nBack):
        sum_of_ck_gam_k = sum_of_ck_gam_k + x[4 + 2 * k] / x[5 + 2 * k]

    return 1.0 * np.append(-x, [x[2] * (rho_iso_inf - 1) + rho_iso_inf * sum_of_ck_gam_k,
                                -x[2] * (rho_iso_sup - 1) - rho_iso_sup * sum_of_ck_gam_k,
                                (rho_yield_inf - 1.) * x[1] - x[2] - sum_of_ck_gam_k,
                                (1. - rho_yield_sup) * x[1] + x[2] + sum_of_ck_gam_k])


def NTR_J_Solver_Lag(f, df, Hf, Lag_arg, tol_k):
    ##### Newton's method with trust region with Steihaug-Toint truncated conjugated gradient method with Jacobi preconditioning ######
    dump_freq = 5  # print x every 5 iterations

    x, Delta, miu, c, g, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup = Lag_arg

    # Initialization

    dk = np.zeros(len(x))

    # Delta = 10
    Tol = tol_k

    eta1 = 0.01
    eta2 = 0.9

    nitNMTRmax = int(1e6)

    gradPhi_fun = gradLag
    HessPhi_fun = hessLag

    # f -> Lagrangian
    gradPhi_test = gradPhi_fun(Lag_arg, df) * 1.  # -> gradLag(Lag_arg, df)
    # -> df = nda.Gradient(errorEnsemble_nda)
    HessPhi_test = HessPhi_fun(Lag_arg, Hf, df) * 1.  # -> hessLag(Lag_arg, Hf, df)
    # -> Hf = nda.Hessian(errorEnsemble_nda)
    reboot = 0

    TOL_Approx = 1e-3  # tolerance for ill-conditioned cases. Accepts the solution as an approximation if the trust-region radius and the scaled step size are too small (TOL_Approx)

    for nit in range(nitNMTRmax):

        print ('\n \n NTR_J It. = ' + str(nit))

        Phi_test_k = f(Lag_arg) * 1.0

        # update constraint values at each iteration

        g = makeG(x, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup)

        # Solve trust-region sub-problem with steihaug-toint method and a scaled hessian

        Diag = np.power(np.abs(np.diag(HessPhi_test)), 0.5)
        Diag[np.abs(Diag) < 1e-10 * np.max(Diag)] = 1
        S = np.diag(1 / Diag)

        # S=np.diag(1/np.power(np.abs(np.diag(HessPhi_test)),0.5))

        HessPhi_test_S = np.dot(np.dot(S.transpose(), HessPhi_test), S)

        gradPhi_test_S = np.dot(S.transpose(), gradPhi_test)

        dk = steihaug(HessPhi_test_S, gradPhi_test_S, Delta)

        # Bring back the scaled step

        dk = np.dot(S, dk)

        # print "\n norm_dk = "+str(np.dot(dk,dk))

        x_trial = x + dk

        model_k = Phi_test_k * 1.0

        model_k1 = Phi_test_k + np.dot(dk, gradPhi_test) + 0.5*np.dot(dk, np.dot(HessPhi_test, dk))

        g_trial = makeG(x_trial, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup)

        Lag_arg_k1 = np.array(
            [x_trial, Delta, miu, c, g_trial, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup]) * 1.0

        Phi_test_k1 = f(Lag_arg_k1) * 1.0

        rho_ = (Phi_test_k - Phi_test_k1) / (model_k - model_k1)

        if ((model_k - model_k1) < 1e-14 and (Phi_test_k - Phi_test_k1) > 0) or np.abs(
                (Phi_test_k - Phi_test_k1) / Phi_test_k) < 1e-14:
            x = x + dk

            g = makeG(x, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup)

            Lag_arg = np.array([x, Delta, miu, c, g, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup]) * 1.0

            gradPhi_test = gradPhi_fun(Lag_arg, df)
            HessPhi_test = HessPhi_fun(Lag_arg, Hf, df)

            if rho_ >= 0.9:
                Delta = 2. * Delta

        if rho_ < eta1 or (Phi_test_k - Phi_test_k1) < 0 or rho_ > 1000:  # too good to be true
            S_inv = np.diag(np.power(np.abs(np.diag(HessPhi_test)), 0.5))
            Delta = 0.5 * np.sqrt(np.dot(np.dot(S_inv, dk), np.dot(S_inv, dk)))

        else:
            x = x + dk

            g = makeG(x, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup)

            Lag_arg = np.array([x, Delta, miu, c, g, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup]) * 1.0

            gradPhi_test = gradPhi_fun(Lag_arg, df)
            HessPhi_test = HessPhi_fun(Lag_arg, Hf, df)

            if rho_ >= 0.9:
                Delta = 2. * Delta

        norm_grad = np.sqrt(np.dot(gradPhi_test, gradPhi_test))

        print ('\n Function: ' + str(Phi_test_k) + ' ; norm_grad: ' + str(norm_grad)) + "  ##"

        nBack = int((len(x) - 4) / 2)

        sum_of_ck_gam_k = 0.0

        for k in range(nBack):
            sum_of_ck_gam_k = sum_of_ck_gam_k + x[4 + 2 * k] / x[5 + 2 * k]

        if norm_grad < Tol:
            break
        elif np.linalg.norm(dk) < TOL_Approx and Delta < TOL_Approx:
            print " WARNING: SECONDARY CONVERGENCE CRITERIA TRIGGERED. NORM OF GRADIENT NOT WITHIN TOLERANCE, THUS ONLY AN APPROXIMATE SOLUTION IS OBTAINED"
            break

        if nit % dump_freq == 0:
            print x

    return Lag_arg_k1


def AugLag_Opt(x, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup, listCleanTests):
    global arrayCleanTests

    arrayCleanTests = list(listCleanTests)

    ## Augmented Lagrangian - cf. Bierlaire 2015 and Bertsekas 2016

    # Initialization

    x = np.array(x) * 1.0

    itLag = 1e3

    TOL = 1e-10
    TOL_Approx = 1e-3  # tolerance for ill-conditioned cases. Accepts the solution as an approximation if the trust-region radius and the scaled step size are too small (TOL_Approx)

    # NTR parameters

    Delta = 10.0

    # Augmented lagrangian parameters

    c = 10.
    eta_hat_zero = 0.1258925
    tau = 10
    alpha = 0.1
    beta = 0.9
    tol_0 = 1. / c
    tol_k = 1. / c
    eta_al = eta_hat_zero / c ** alpha

    # Correction of initial point for feasible dual solution

    nBack = int((len(x) - 4) / 2)

    rho_iso_start = (rho_iso_inf + rho_iso_sup) / 2.
    rho_yield_start = (rho_yield_inf + rho_yield_sup) / 2.

    x[2] = -(1 - rho_yield_start) / (1 - (rho_iso_start - 1) / rho_iso_start) * x[1]

    sum_of_ck_gam_k = -(rho_iso_start - 1) / rho_iso_start * x[2]

    for k in range(nBack):
        x[4 + 2 * k] = sum_of_ck_gam_k / (nBack * 1.)
        x[5 + 2 * k] = 1.

    # Initial inequality function vector

    g = makeG(x, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup)

    # Inequality multipliers

    miu = np.zeros(len(g))

    # Defining gradient and Hessian of errors function with AlgoPy

    grad_error_fun = nda.Gradient(errorEnsemble_nda)

    Hess_error_fun = nda.Hessian(errorEnsemble_nda)

    approxIt = 0
    approxIt_lim = 10  # only accept a limited number of ill-conditioned iterations.

    for i in range(int(itLag)):

        print("##########      New Lagrangian Step      ###########")

        x_1 = x * 1.0

        Larg = [x, Delta, miu, c, g, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup]

        Larg = NTR_J_Solver_Lag(Lagrangian, grad_error_fun, Hess_error_fun, Larg, tol_k)

        x, Delta, miu, c, g, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup = Larg

        sum_of_ck_gam_k = 0.0

        for k in range(nBack):
            sum_of_ck_gam_k = sum_of_ck_gam_k + x[4 + 2 * k] / x[5 + 2 * k]

        g = makeG(x, rho_iso_inf, rho_iso_sup, rho_yield_inf, rho_yield_sup)

        g_max = np.maximum(np.zeros(len(g)), g)
        norm_ineq = np.sqrt(np.dot(g_max, g_max))

        if norm_ineq <= eta_al:

            for j in range(len(miu)):
                miu[j] = max(0, miu[j] + c * g[j])

            tol_k = tol_k / c
            eta_al = eta_al / c ** beta

        else:

            c = tau * c
            tol_k = tol_0 / c
            eta_al = eta_hat_zero / c ** alpha

        d = (x - x_1) * 1.0

        gradLag_vec = gradLag(Larg, grad_error_fun)

        # print ("Norm gradLag=", np.sqrt(np.dot(gradLag_vec, gradLag_vec)), " ,Norm_ineq=", norm_ineq)

        if np.sqrt(np.dot(gradLag_vec, gradLag_vec)) < TOL and norm_ineq < TOL:
            print "####################################################"
            print "### SUCCESSFUL AUGMENTED LAGRANGIAN OPTIMIZATION ###"
            print "####################################################"
            print "########## TERMINATING AUGMENTED LAGRANGIAN ########"
            print "####################################################"
            print ("x = ", x)
            break
        elif np.linalg.norm(d) < TOL_Approx and Delta < TOL_Approx:
            print " WARNING: SECONDARY CONVERGENCE CRITERIA TRIGGERED. NORM OF GRADIENT NOT WITHIN TOLERANCE."
            print " ONLY AN APPROXIMATE SOLUTION IS OBTAINED"
            approxIt = approxIt + 1
            Delta = 1.0
            if approxIt > approxIt_lim:
                print "####################################"
                print "# TERMINATING AUGMENTED LAGRANGIAN #"
                print "####################################"
                print ("x = ", x)
                break

            else:
                continue

    return x
