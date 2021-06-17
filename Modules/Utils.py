import csv
import os, subprocess
import random


def make_dir(path):
    """recursively create the directory specified by path if it does not exist

    Args:
        path: path to the directory that will be created

    Returns:
        str: path, identical to the input argument path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    assert os.path.exists(path), "failed to create {}".format(path)
    return path


def run(command, fault_tolerant=False):
    """use the subprocess.run function to run a command

    Args:
        command: list of strings to be passed as the first argument of subprocess.run()
        fault_tolerant: if False (default) and the command fails to run, raise an exception and halt the script

    Returns:
        str: stdout from executing subprocess.run(command)

    Raises:
        Exception: if subprocess.run() produces a nonzero return code and fault_tolerant is False
    """

    output = subprocess.run(command, stdout=subprocess.PIPE, encoding='utf-8')
    if output.returncode != 0:
        if not fault_tolerant:
            print(output.stderr)
            raise Exception('error running the following command: {}'.format(' '.join(command)))
        else:
            print('error running the following command: {}'.format(' '.join(command)))
            print('fault tolerant set to True, ignoring error')
    return output.stdout


def xyminmax_to_xywh(xmin, ymin, xmax, ymax):
    """convert box coordinates from (xmin, ymin, xmax, ymax) form to (x, y, w , h) form"""
    return [xmin, ymin, xmax - xmin, ymax - ymin]


def xywh_to_xyminmax(x, y, w, h):
    """convert box coordinates from (x, y, w , h) form to (xmin, ymin, xmax, ymax) form"""
    return [x, y, x + w, y + h]


class AverageMeter(object):
    """Computes and stores the running average of a metric. Useful for updating metrics after each epoch / batch."""

    def __init__(self):
        """default all values to upon class declaration."""
        self._reset()

    def _reset(self):
        """reset all metrics to 0 when initiating."""
        self.val = 0    #: current value
        self.avg = 0    #: running average of metric
        self.sum = 0    #: running sum of metric
        self.count = 0  #: running count of metric

    def update(self, val, n=1):
        """Update the current value, as well as the running sum, count, and average.

        Args:
            val (float): value used to update metrics
            n (int): number of instances with the value val. Default 1

        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    """manages creation of logfiles that track basic training/evaluation stats."""

    def __init__(self, path, header):
        """open the logfile and write its header.

        Args:
            path (str): path to the logfile
            header (list of str): column names
        """
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        """close the logfile."""
        self.log_file.close()

    def log(self, values):
        """write a new row to the logfile.

        Args:
            values (dict): dictionary of key-value pairs, where each key must be a column name in self.header
        """
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()
