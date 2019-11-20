"""@package reduced_cg
Reduced conjugate gradient solver for the trust region problem with linear constraints.
"""
import numpy as np
import numpy.linalg as la
from model_minimizer import trust_region_intersect, model_minimizer


def reduced_cg(g, c, a, b, delta):
    """ Solves a quadratic problem subjected to linear and trust-region constraints.

    :param np.array g: (n, n) Matrix in quadratic problem.
    :param np.array c: (n, 1) Vector in quadratic problem.
    :param np.array a: (m, n) Linear coefficients of x in equality constraints, m constraints are specified, m < n.
    :param np.array b: (m, 1) Vector of constants in equality constraints.
    :param float delta: Trust-region radius.
    :return np.array: (n, 1) Trajectory of x that minimizes the quadratic problem.


    This function is based on Algorithm 16.2 from [1], pg. 461. An additional trust-region constraint is added to
    the algorithm based on the 2-norm of the x_z vector. The original quadratic problem to solve is

        minimize x^T . c  +  1/2 x^T . G . x
    (P1)    x
        subject to A . x  =  b,
                   ||x||_2  <=  Delta

    The equivalent reduced problem solved by this function is

        minimize x^T . c_z  +  1/2 x_z^T . Z^T . G . Z . x_z
    (P2)    x_z
        subject to ||x||_2  <=  Delta

    For the trust-region SQP optimization problem, the parameters are defined for the k'th increment in (P1) as:
        - The parameters are defined as follows: [reduced_cg()] = [(P1)] = [SQP problem]
        - g = G = hess_xx[L(x_k)]
        - c = c = grad_x[f(x_k)]
        - a = A = jacobian[g(x_k)]
        - b = b = -g*(x_k)  <-- notice the minus sign
    where L is the Lagrangian, f is the objective function, and g* are the equality constraints formed from inequality
    constraints using slack variables. The transformation from (P1) -> (P2) is handled by this function internally. An
    additional guard is added when negative curvature is encountered.

    If the constraints A.x == b and ||x|| < Delta are infeasible, then a relaxation is calculated (Byrd-Omojokun
    approach). The relaxation is calculated by solving a trust-region subproblem nearly exactly using the
    model-minimizer method.

    References:
        [1] Nocedal and Wright (2006) "Numerical optimization"
        [2] Gould et al. (2001) "On the solution of equality constrained quadratic programming problems arising in
            optimization"
        [3] Bierlaire (2015) "Optimization: Principles and Algorithms"
        [4] Conn et al. (2000) "Trust region methods"
    """
    tol = np.sqrt(np.finfo(float).eps)
    radius_reduction_factor = 0.8  # necessary to allow some "slack" in the constraint to reduce the objective function
    m, n = np.shape(a)
    max_iter = n - m + 1

    # Calculate the orthonormal range and null-space of A^T
    q, _ = la.qr(a.transpose(), mode='complete')
    y = q[:, :m]  # basis for the range of A^T
    z = q[:, m:]  # basis for the null-space of A^T

    # Check that the x_y component of the solution is within the trust region
    x_y = la.solve(np.matmul(a, y), b)  # x_y satisfies the constraint A.x == b
    x_sol_prev = np.matmul(y, x_y)
    if la.norm(x_sol_prev, 2) > delta * radius_reduction_factor:
        # The constraints are not feasible with the trust-region, so calculate a relaxation and continue
        # Choose r = a.v - b => then we solve min_x x.g.x + 1/2 x.c ; s.t a.x == r --> replace b with r
        print 'Applying a relaxation to the constraint.'
        v = solve_normal_step(a, -b, radius_reduction_factor * delta)
        r = np.matmul(a, v)  # relaxation
        x_y = la.solve(np.matmul(a, y), r)
        x_sol_prev = np.matmul(y, x_y)
        if la.norm(x_sol_prev, 2) > delta:
            # Still not feasible, something went wrong
            raise ValueError('Trust-region is incompatible with the constraints')

    # Initialize values for the algorithm
    h = np.diag(np.abs(np.diag(g)))  # try Jacobi preconditioner
    h[h < 1.e-10 * np.max(h)] = 1.
    w_zz = np.matmul(z.transpose(), np.matmul(h, z))
    w_zz_inv = la.inv(w_zz)
    c_z = reduce(np.dot, [z.transpose(), g, y, x_y]) + np.matmul(z.transpose(), c)
    x_z = np.zeros((n - m, 1))  # necessary to start at 0 to find the trust-region intersection, see ref. [2]
    r_z = reduce(np.dot, [z.transpose(), g, z, x_z]) + c_z
    g_z = np.matmul(w_zz_inv, r_z)
    d_z = -1.0 * g_z

    convergence_criteria = float(np.abs(np.dot(r_z.transpose(), np.dot(w_zz_inv, r_z))))
    iteration = 0
    while convergence_criteria > tol and iteration < max_iter:
        # Calculate the step length
        alpha = float(np.dot(r_z.transpose(), g_z) / reduce(np.dot, [d_z.transpose(), z.transpose(), g, z, d_z]))

        # Test for negative curvature
        if reduce(np.dot, [d_z.transpose(), z.transpose(), g, z, d_z]) < 0.:
            # Go to the boundary of the trust-region
            # For method, see ref. [3] pg. 298
            step_factor = trust_region_intersect(x_sol_prev, np.matmul(z, alpha * d_z), delta)
            x_z = x_z + step_factor * alpha * d_z
            break

        # Test the trust-region constraint
        x_sol_trial = x_sol_prev + np.matmul(z, alpha * d_z)
        if la.norm(x_sol_trial, 2) >= delta:
            # We (miraculously) hit the boundary of the trust-region
            if la.norm(x_sol_trial, 2) == delta:
                x_z = x_z + alpha * d_z
                break
            else:
                # Find the intersection with the trust-region between the previous point and the trial point
                step_factor = trust_region_intersect(x_sol_prev, np.matmul(z, alpha * d_z), delta)
                x_z = x_z + step_factor * alpha * d_z
                break

        # OK, calculate the next CG iteration
        x_z = x_z + alpha * d_z
        x_sol_prev = x_sol_trial
        r_z_p = r_z + alpha * reduce(np.dot, [z.transpose(), g, z, d_z])
        g_z_p = np.matmul(w_zz_inv, r_z_p)
        beta = np.dot(r_z_p.transpose(), g_z_p) / np.dot(r_z.transpose(), g_z)
        d_z = -1. * g_z_p + beta * d_z
        g_z = g_z_p
        r_z = r_z_p

        convergence_criteria = float(np.abs(np.dot(r_z.transpose(), np.dot(w_zz_inv, r_z))))
        iteration = iteration + 1

    # Get the solution point, recalculate to minimize any round-off errors
    # Total step considers the x_y step to satisfy the constraints, and the x_z step to minimize the obj. fun
    x_sol = np.matmul(y, x_y) + np.matmul(z, x_z)

    # Calculate the Lagrange multipliers, see [1] pg. 538
    lam_sol = la.solve(np.matmul(a, y).transpose(), np.matmul(y.transpose(), c + np.matmul(g, x_sol)))

    return [x_sol, lam_sol]


def solve_normal_step(a, c, radius):
    """ Returns the solution to the normal subproblem.

    :param np.array a: (m, n) Jacobian of the constraint function.
    :param np.array c: (m, 1) Constraint function values.
    :param float radius: Trust-region radius.
    :return np.array : (n, 1) Solution to the normal subproblem.

    The normal subproblem is defined as

    minimize ||A_k . v  +  c_k||_2^2
        v
    subject to ||v||_2 <= Delta

    where A_k is the Jacobian of the constraints, c_k is the constraint function, and Delta is the trust-region
    radius. The model minimizer method is used to solve this trust-region problem. See [4] pg. 547 for details.
    """
    h = np.matmul(a.transpose(), a)
    g = np.matmul(a.transpose(), c)
    d_x = model_minimizer(h, g, radius)
    return d_x
