import logging
from logging.handlers import TimedRotatingFileHandler


class Logger:
    """logger class"""
    def __init__(self, log_file_name, log_level, logger_name):
        """creat logger"""
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)

        file_handler = TimedRotatingFileHandler(filename=log_file_name, when='S', interval=1,
                                                backupCount=3)
        file_handler.suffix = "%Y-%m-%d"
        console_handler = logging.StreamHandler()

        # 定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 给logger添加handler
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger
