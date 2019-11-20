"""@package scipy_dumper
Dumper for the scipy optimization methods.
"""
import numpy as np


class ScipyBasicDumper:
    def __init__(self, x_file, fun_file):
        """ Constructor.

        :param str x_file: Path to the file to write the primal variable history.
        :param str fun_file: Path the the file to write the objective function history.

        Notes:
            - The constructor clears the files.
        """
        self.dump_file = x_file
        self.function_file = fun_file

        with open(self.dump_file, 'w') as f:
            # Clear the file
            pass

        with open(self.function_file, 'w') as f:
            f.write('iteration, function, norm_grad_Lagr\n')

    def dump(self, x, state):
        it_num = state.niter
        f_val = state.fun
        norm_grad_lag = state.optimality
        with open(self.dump_file, 'ab') as fi:
            np.savetxt(fi, x.reshape((1, len(x))), fmt='%7.6e')
        with open(self.function_file, 'ab') as fi:
            fi.write('{0}, {1:5.4e}, {2:5.4e}\n'.format(it_num, f_val, norm_grad_lag))
