"""@package reciprocal_barrier
Implements the reciprocal barrier function.
"""

from barrier_function import BarrierFunction


class ReciprocalBarrier(BarrierFunction):
    def __init__(self, initial_height=1., decrease_rate=1.):
        """ Defines the reciprocal barrier function.

        :param float initial_height: Initial value of the barrier parameter.
        :param float decrease_rate: Controls the rate of decrease for the barrier parameter, must be > 1.0 for
            convergence.

        Assuming that the barrier parameter is greater than 0.0, the returned value -> 0 as the number of
        iterations -> inf.

        See: Conn, Gould, and Toint (2000), "Trust Region Methods", Chapter 13.
        """
        BarrierFunction.__init__(self, initial_height, decrease_rate)
        return

    def value(self, xi):
        return self.barrier_parameter / xi ** 2

    def update_barrier_parameter(self):
        self.barrier_parameter = self.initial_height * (1. / self.decrease_rate) ** self.iteration
        self.iteration += 1
        return
