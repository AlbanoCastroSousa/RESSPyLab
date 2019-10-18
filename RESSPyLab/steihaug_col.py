"""@package steihaug_col
Steihaug-Toint CG method with column vector ordered input and output.
"""
import numpy as np


def steihaug_col(Q, b, Delta):
    """ Steihaug-Toint truncated conjugated gradient method.

    :param np.array Q: left-hand side matrix
    :param np.array b: right-hand side column vector
    :param float Delta: Trust region radius
    :return np.array: x, the solution to Qx = b, as a column vector.
    """
    TOL = 1e-10
    x_1 = np.zeros(np.shape(b))
    d = -b * 1.0

    flag = 0
    for i in range(len(b) + 1):
        x_prev = x_1
        if np.dot(d.transpose(), np.dot(Q, d)) < 0:
            # Hessian is not positive definite in this direction, go to trust-region boundary
            flag = 1
            a_ = np.dot(d.transpose(), d)
            b_ = np.dot(2 * x_1.transpose(), d)
            c_ = np.dot(x_1.transpose(), x_1) - Delta ** 2
            lambda_ = (-b_ + np.sqrt(b_ ** 2 - 4 * a_ * c_)) / (2 * a_)
            x_stei = x_1 + lambda_ * d
            break

        # OK, take a step
        alpha = -np.dot(d.transpose(), np.dot(Q, x_1) + b) / np.dot(d.transpose(), np.dot(Q, d))
        x_1 = x_prev + alpha * d

        if np.linalg.norm(x_1) > Delta:
            # Outside the trust-region, go back to boundary
            flag = 1
            a_ = np.dot(d.transpose(), d)
            b_ = np.dot(2 * x_prev.transpose(), d)
            c_ = np.dot(x_prev.transpose(), x_prev) - Delta ** 2
            lambda_ = (-b_ + np.sqrt(b_ ** 2 - 4 * a_ * c_)) / (2 * a_)
            x_stei = x_prev + lambda_ * d
            break

        beta = np.linalg.norm(np.dot(Q, x_1) + b) ** 2 / np.linalg.norm(np.dot(Q, x_prev) + b) ** 2
        d = -np.dot(Q, x_1) - b + beta * d

        if np.linalg.norm(np.dot(Q, x_1) + b) < TOL:
            break

    if flag == 0:
        x_stei = x_1

    # print "Steihaug its = ", i

    return x_stei
