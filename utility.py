
import os
import errno
import sys
import logging


class StreamToLogger(object):
    """
    Redirect all the stdout/err to the logger, therefore both print and traceback
    are redirected to logger
    """

    def __init__(self, logger, LogFile='test', log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
        self.logFile = LogFile

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            filename=self.logFile,
            filemode='a'
        )
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


# utility function
def makedir(path):  # se esiste già non fa nulla e salta l'exceprtion
    try:
        os.makedirs(path)
        print("make " + path + " dir")
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        pass


# def initLogger(id, log):
#     '''
#     Init the logger for the experiment: other module should get the same logger using the id of experiment
#     :param id: is the identifier of the experiment, here used as the name of the logger
#     :param log: il flag: if true save log in file or print in active console otherwise
#     :return: return the logger
#     '''
#     import logging
#
#     logFolder = 'logs'
#     nameFileLog = os.path.join(logFolder, 'process_' + str(id) + '.log')
#     logger = logging.getLogger(str(id))
#     logger.setLevel(logging.DEBUG)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
#     if log:
#         makedir(logFolder)  # crea la fold solo se non esiste
#         if os.path.isfile(nameFileLog):  # if there is a old log, save it with another name
#             os.rename(nameFileLog, nameFileLog + '_' + str(len(os.listdir(logFolder)) + 1))  # so the name is different
#         # create file handler which logs even debug messages
#         fh = logging.FileHandler(nameFileLog)
#         fh.setLevel(logging.DEBUG)
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)
#
#     else:
#         # create console handler with a higher log level
#         ch = logging.StreamHandler()
#         ch.setLevel(logging.DEBUG)
#         ch.setFormatter(formatter)
#         logger.addHandler(ch)
#
#     return logger

