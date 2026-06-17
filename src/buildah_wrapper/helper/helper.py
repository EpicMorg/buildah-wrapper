import logging
import colorlog

from buildah_wrapper.settings import settings


def setup_logger():
    logger = logging.getLogger("logger_buildah")
    logger.setLevel(settings.log_level.upper())

    ch = colorlog.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = colorlog.ColoredFormatter(
        fmt=settings.log_format,
        datefmt=settings.datefmt,
        log_colors=settings.log_colors,
    )

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger