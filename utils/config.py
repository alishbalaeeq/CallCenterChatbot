import os
from loguru import logger

logfile = "files/output.log"
logger.add(logfile, colorize=True, enqueue=True)