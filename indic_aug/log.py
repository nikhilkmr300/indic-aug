import logging

# Setting up logger.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('indic_aug')

# Number of dashes in dashed line separating logging output between docs.
NUM_LOGGER_DASHES = 30

def change_logpath(logpath):
    """The default path for logging output is ``stdout``. Use this function to
    redirect logging output to a logfile at ``logpath``.

    :param logpath: Path to logfile.
    :type logpath: str
    """

    new_handler = logging.FileHandler(logpath, 'w')
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(new_handler)