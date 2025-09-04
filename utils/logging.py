import sys

from loguru import logger

logger.remove()
logger.level("INFO", color="<green>")

logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:MM-DD HH:mm:ss}</green> | <level>{level:6.6}</level> | <cyan>{module}</cyan>:<cyan>{line:3}</cyan>: {message}",
    # format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
    backtrace=False,
)
