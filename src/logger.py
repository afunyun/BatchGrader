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
from src.config_loader import load_config
from rich.logging import RichHandler

SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

class BatchGraderLogger:
    def __init__(self, log_dir='output/logs'):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'batchgrader_run_{timestamp}.log')
        self.logger = logging.getLogger(f'batchgrader_{timestamp}')
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

    def info(self, msg):
        self.logger.info(msg)
    def success(self, msg):
        self.logger.log(SUCCESS_LEVEL_NUM, msg)
    def warning(self, msg):
        self.logger.warning(msg)
    def error(self, msg):
        self.logger.error(msg)
    def event(self, msg):
        self.logger.info(f"[EVENT] {msg}")
    def debug(self, msg):
        self.logger.debug(msg)

logger = BatchGraderLogger()
