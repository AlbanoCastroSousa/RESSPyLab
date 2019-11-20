"""@package basic_dumper
Object to periodically write results to screen and file.
"""
import numpy as np

from gensolver_dumper import GenSolverDumper


class BasicDumper(GenSolverDumper):

    def __init__(self, output_file, numpy_printopts=None, verbose_dump_freq=5, function_output_file=''):
        """ Outputs basic convergence info to screen, and optionally saves the primal variables to a file.

        :param str output_file: Path to the file to store the primal variable output, if output_file == '' then will
            not save to file.
        :param dict numpy_printopts: Key / value pairs for the np.set_printoptions function applied to all screen output.
        :param int verbose_dump_freq: Prints advanced info, and saves primals, at specified increment intervals.
        :param str function_output_file: Path to the file to store the objective function value

        - Values are saved to file every [verbose_dump_freq] iterations of the procedure.
        """
        GenSolverDumper.__init__(self, output_file)
        self.verbose_dump_freq = verbose_dump_freq
        if numpy_printopts is not None:
            np.set_printoptions(**numpy_printopts)

        self.function_file = function_output_file
        if self.function_file != '':
            with open(self.function_file, 'w') as f:
                # clear the file if it already exists and write the header
                f.write('iteration, function, norm_grad_Lagr\n')

    def dump(self, dump_info):
        """ Dumps the parameters in dump_info.

        :param dict dump_info: Contains the parameters to print to screen and save. The keys are of type str.

        - dump_info must contain the following keys:
            - 'x'
            - 'f_val'
            - 'norm_grad_lag'
            - 'it_num'
        - If self.dump_file != '' the value of x is appended to the end of self.dump_file.
        - If self.function_file != '' the iteration number, function value, and gradient of the Lagrangian are
            appended to the end of self.function_file
        """
        # Output basic info to screen
        it_num = dump_info['it_num']
        f_val = dump_info['f_val']
        norm_grad_lag = dump_info['norm_grad_lag']
        basic_dump_str = "\nIt. = {0}:\tf(x) = {1:e} ; ||grad[L]|| = {2:e}"
        print basic_dump_str.format(it_num, f_val, norm_grad_lag)

        # Output more advanced info at specified increments
        if dump_info['it_num'] % self.verbose_dump_freq == 0:
            # Remove the values we just displayed
            for key in ['f_val', 'it_num', 'norm_grad_lag']:
                del dump_info[key]
            # Now display the rest
            for key, value in dump_info.items():
                if type(value) is np.ndarray:
                    print key + ' = ', value.reshape(-1)
                else:
                    print key + ' = ', value

            # Save x values
            if self.dump_file != '':
                x = dump_info['x']
                with open(self.dump_file, 'ab') as f:
                    np.savetxt(f, x.reshape((1, len(x))), fmt='%7.6e')
            # Save the iteration, function value, and norm of the gradient of the Lagrangian
            if self.function_file != '':
                with open(self.function_file, 'ab') as f:
                    f.write('{0}, {1:5.4e}, {2:5.4e}\n'.format(it_num, f_val, norm_grad_lag))
        return
