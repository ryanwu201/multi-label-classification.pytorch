import sys
from datetime import datetime
from io import TextIOBase


class LoggerFileWrapper(TextIOBase):
    """
    sys.stdout = _LoggerFileWrapper(logger_file_path)
    Log with PRINT Imported from NNI
    """

    def __init__(self, logger_file_path):
        self.terminal = sys.stdout
        logger_file = open(logger_file_path, 'a')
        self.file = logger_file

    def write(self, s):
        self.terminal.write(s)
        if s != '\n':
            _time_format = '%m/%d/%Y, %I:%M:%S %p'
            cur_time = datetime.now().strftime(_time_format)
            self.file.write('[{}] PRINT '.format(cur_time) + s + '\n')
            self.file.flush()
        return len(s)
