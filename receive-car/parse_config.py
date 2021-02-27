import os
import logging
from pathlib import Path
from datetime import datetime
from logger import setup_logging


class ConfigParser:
    def __init__(self, log_path, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        run_id = None
        if run_id is None:
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')

        save_dir = Path(log_path)
        log_dir = save_dir / run_id
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        # self.img_dir = save_dir / 'img' / run_id
        # if not self.img_dir.exists():
        #     self.img_dir.mkdir(parents=True)
        setup_logging(log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger
    def get_imgdir(self,):
        return self.img_dir

if __name__ == "__main__":
    my_logger = ConfigParser()
    logger = my_logger.get_logger("test", 2)
    logger.info("123")
