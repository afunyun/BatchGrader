"""
BatchGraderLogger: Dual-handler logger with RichHandler for console and FileHandler for persistent logs.
- Console output uses rich.logging.RichHandler for colorized, pretty logs.
- File logs are written with standard formatting for post-mortem/debug.
- API: logger.info, logger.success, logger.warning, logger.error, logger.event, logger.debug
"""
import sys
import os
import logging
from datetime import datetime
from rich.logging import RichHandler

SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

class BatchGraderLogger:
    def __init__(self, log_dir='output/logs'):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'batchgrader_run_{timestamp}.log')
        
        self.logger = logging.getLogger('BatchGrader')

        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.INFO)
            
            console_handler = RichHandler(show_time=True, show_level=True, show_path=False, rich_tracebacks=True)
            console_handler.setLevel(logging.INFO)
            
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.INFO)
            
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
            self.info(f"BatchGrader run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def get_logger(self):
        return self.logger

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    def success(self, msg, *args, **kwargs):
        self.logger.log(SUCCESS_LEVEL_NUM, msg, *args, **kwargs)
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    def event(self, msg, *args, **kwargs):
        self.logger.info(f"[EVENT] {msg}", *args, **kwargs)
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

logger = BatchGraderLogger()
