"""@package model_minimizer
Model minimizer method to solve the trust-region subproblem.
"""
import numpy as np
from numpy import linalg as la

from posdef_check import posdef_check

TOL = np.sqrt(np.finfo(float).eps)


def h_lambda(evals, evecs, lam):
    """ Returns the positive definite H by adding a factor of the left-most eigenvalue.

    H(\lambda) = Q^T (H_bar + \lambda I) Q

    :param np.array evals: (n,) Eigenvalues.
    :param np.array evecs: (n, n) Eigenvectors, Q.
    :param float lam: Factor to multiply to the identity matrix.
    :return np.array: positive definite matrix.
    """
    n, _ = np.shape(evecs)
    i = np.identity(int(n))
    h_bar = np.diag(evals) + lam * i
    h_lam = np.matmul(evecs.transpose(), np.matmul(h_bar, evecs))
    return h_lam


def eig_decomp(h):
    """ Returns the elements of the eigendecomposition and the minimum eigenvalue/vector for the model minimizer.

    The eigendecomposition is defined as h = Q^T \Lambda Q

    :param np.array h: (n, n) Matrix to be decomposed.
    :return list:
        - evals np.array: (n, ) Unsorted eigenvalues of h, \Lambda = diag(evals)
        - evecs np.array: (n, n) Unsorted eigenvectors of h, provides the matrix Q
        - left_eval float: A value slightly less than the minimum eigenvalue
        - left_evec np.array: (n, 1) Eigenvector corresponding to the minimum eigenvalue
    """
    [evals, evecs] = la.eig(h)
    i = np.argmin(evals)
    left_eval = evals[i]
    evecs = evecs.transpose()
    left_evec = evecs[:, i]

    slightly_less_factor = (1. + np.abs(left_eval)) * TOL  # to make the Hessian positive definite
    left_eval = left_eval - slightly_less_factor
    return [evals, evecs, left_eval, left_evec]


def trust_region_intersect(d_c, d, Delta):
    """ Returns the factor such that the search direction reaches the edge of the trust radius.

    :param np.array d_c: (n, 1) Starting point.
    :param np.array d: (n, 1) Direction.
    :param float Delta: Trust-region radius.
    :return float: Minimum multiplier lam, such that ||s + lam * d||_2 = Delta.

    - s and d are assumed to be column vectors.
    """
    a = np.dot(d.transpose(), d)
    b = 2 * np.dot(d_c.transpose(), d)
    c = np.dot(d_c.transpose(), d_c) - Delta ** 2
    lam = (-b + np.sqrt(b ** 2 - 4. * a * c)) / (2 * a)
    return float(lam)


def model_minimizer(h, g, Delta):
    """ Returns the step to the current model minimizer point.

    :param np.array h: Hessian matrix.
    :param np.array g: Gradient column vector.
    :param float Delta: Trust-region radius.
    :return np.array: Step to the model minimizer.


    We are solving the minimization problem of a quadratic function subjected to a trust-region constraint

    minimize q(s) = g^T . s  +  1/2 * s^T . h . s
        s
    subject to ||s||_2 <= Delta

    where g = grad[f(x)], h = hess[f(x)], and Delta is the trust-region radius. The model minimizer method finds the
    nearly exact minimum of this problem using an eigendecomposition. This function is based on Alg 7.3.6 [1], pg. 199.
    This method should only be used when the eigendecomposition of h is cheap (i.e,. h is small or tridiagonal), and
    additionally note that the implementation below is likely not very efficient at the moment.

    References:
        [1] Conn et al. (2000) "Trust Region Methods"
    """
    k_easy = 0.001  # we don't need to solve the problem TOO precisely since the model itself is an approx. in any case
    maximum_iterations = 25
    stop = False

    # Step 1
    [evals, evecs, lam_1, u_1] = eig_decomp(h)
    if lam_1 >= 0:
        # Positive definite Hessian
        lam = 0.
        h_mod = h_lambda(evals, evecs, 0.)
    else:
        # Indefinite Hessian
        lam = -lam_1
        h_mod = h_lambda(evals, evecs, lam)

    # Step 2
    s = la.solve(h_mod, -g.reshape(-1, 1))

    # Step 3
    norm_s = la.norm(s)
    if norm_s <= Delta:
        if lam == 0. or np.abs(norm_s - Delta) <= TOL:
            stop = True
        else:
            # Hard case
            alpha = trust_region_intersect(s, u_1, Delta)
            s = s + alpha * u_1.reshape(-1, 1)
            stop = True

    # Step 4
    iteration_number = 0
    while stop is False and np.abs(norm_s - Delta) > k_easy * Delta and iteration_number < maximum_iterations:
        # Easy case
        iteration_number += 1

        # Step 5
        l_chol = la.cholesky(h_mod)
        s = la.solve(h_mod, -g)
        w = la.solve(l_chol, s)

        norm_s = la.norm(s)
        lam = lam + (norm_s - Delta) / Delta * (norm_s / la.norm(w)) ** 2
        h_mod = h_lambda(evals, evecs, lam)
        if not posdef_check(h_mod) or lam < -lam_1:
            [_, _, lam_check, _] = eig_decomp(h_mod)
            raise ValueError("lam = {0}, lam_min = {1}, lam_1 = {2}, phi(lam) = {2}".format(lam, lam_check, lam_1,
                                                                                            1 / norm_s - 1 / Delta))

    # print "Model min. its = {0}".format(iteration_number)
    return s
