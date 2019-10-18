"""@package modified_cholesky
Function to perform the modified Cholesky decomposition.
"""
import numpy as np
import numpy.linalg as la


def modified_cholesky(a):
    """ Returns the matrix A if A is positive definite, or returns a modified A that is positive definite.

    :param np.array a: (n, n) The symmetric matrix, A.
    :return list: [np.array (n, n), float] Positive definite matrix, and the factor required to do so.

    See Bierlaire (2015) Alg. 11.7, pg. 278.
    """
    iteration = 0
    maximum_iterations = 10
    identity = np.identity(len(a))
    a_mod = a * 1.0
    identity_factor = 0.

    successful = False
    while not successful and iteration < maximum_iterations:
        try:
            la.cholesky(a_mod)
            successful = True
        except la.LinAlgError:
            identity_factor = np.max([2 * identity_factor, 0.5 * la.norm(a, 'fro')])
            a_mod = a + identity_factor * identity
    return [a_mod, identity_factor]
