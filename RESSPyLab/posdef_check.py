"""@package posdef_check
Check if a matrix is positive definite.
"""
import numpy as np


def posdef_check(a):
    """ Test if a is positive-definite matrix.

    :param np.array a: An nxn matrix.
    :return bool: True if a is positive-definite, False otherwise.
    """
    try:
        np.linalg.cholesky(a)
        posdef = True
    except np.linalg.LinAlgError:
        posdef = False
    return posdef
