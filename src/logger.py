import os
import logging


def get_logger(name: str = "TASK_2") -> logging.Logger:
    """Return a configured logger consistent with the notebook's setup."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        os_dir = "logs"
        os.makedirs(os_dir, exist_ok=True)

        file_handler = logging.FileHandler(
            os.path.join(os_dir, "task2.log")
        )
        console_handler = logging.StreamHandler()

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
