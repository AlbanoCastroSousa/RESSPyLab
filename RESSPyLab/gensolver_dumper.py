"""@package gensolver_dumper
Abstract class for dumpers.
"""


class GenSolverDumper:

    def __init__(self, output_file):
        """ Abstract class for creating dumper objects.

        :param str output_file: Path to the file to store output, if output_file == '' then will not output to file.
        """
        self.dump_file = output_file
        if self.dump_file != '':
            f = open(self.dump_file, 'w')  # clear the file if it already exists
            f.close()
        return

    def dump(self, dump_info):
        raise Exception("not implemented in {0}".format(self))
