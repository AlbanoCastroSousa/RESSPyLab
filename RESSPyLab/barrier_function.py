"""@package barrier_function
Object that applies the barrier function to the augmented Lagrangian solver.
"""


class BarrierFunction:
    def __init__(self, initial_height, decrease_rate):
        """ Abstract class to define barrier functions for use with AugLagGenSolver.

        :param float initial_height: Initial barrier value.
        :param float decrease_rate: Controls the rate of decrease so that value -> 0 as iterations -> inf
        """
        self.iteration = 0
        self.initial_height = initial_height
        self.barrier_parameter = initial_height  # barrier_parameter = \mu_k in Conn et al. (2000)
        self.decrease_rate = decrease_rate
        return

    def value(self, xi):
        """ Value of the barrier function for x_i. """
        raise Exception("not implemented in {0}".format(self))

    def update_barrier_parameter(self):
        """ Controls the update of the barrier parameter. """
        raise Exception("not implemented in {0}".format(self))
