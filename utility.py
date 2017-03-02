
import logging
import os
import errno


class MyLogger:
    def __init__(self, id, logToFile):
        logFolder = 'logs'
        nameFileLog = os.path.join(logFolder, 'process_' + str(id) + '.log')
        logger = logging.getLogger('experiment')
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if logToFile:
            makedir(logFolder)  # crea la fold solo se non esiste
            if os.path.isfile(nameFileLog):  # if there is a old log, save it with another name
                os.rename(nameFileLog, nameFileLog + '_' + str(len(os.listdir(logFolder)) + 1))  # so the name is different
            # create file handler which logs even debug messages
            fh = logging.FileHandler(nameFileLog)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        else:
            # create console handler with a higher log level
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        self.logger = logger

    def debug(self, messageToLog):
        self.logger.debug(messageToLog)

    def info(self, messageToLog):
        self.logger.info(messageToLog)

    def warning(self, messageToLog):
        self.logger.warning(messageToLog)

    def error(self, messageToLog):
        self.logger.error(messageToLog)



# utility function
def makedir(path):  # se esiste già non fa nulla e salta l'exceprtion
    try:
        os.makedirs(path)
        print("make " + path + " dir")
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        pass