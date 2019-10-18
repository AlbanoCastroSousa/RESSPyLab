"""@package log_barrier
Implements the logarithmic barrier function.
"""
import numpy as np

from barrier_function import BarrierFunction


class LogBarrier(BarrierFunction):
    def __init__(self, initial_height=100., decrease_rate=1.2):
        """ Defines the log barrier function.

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
        return -self.barrier_parameter * np.log(xi)

    def update_barrier_parameter(self):
        self.barrier_parameter = self.initial_height * (1. / self.decrease_rate) ** self.iteration
        self.iteration += 1
        return
