import sys
import os
import logging
from datetime import datetime
from colorama import init, Fore, Style

init(autoreset=True)

LOG_COLORS = {
    'INFO': Fore.CYAN,
    'SUCCESS': Fore.GREEN,
    'WARNING': Fore.YELLOW,
    'ERROR': Fore.RED,
    'EVENT': Fore.MAGENTA,
    'DEBUG': Fore.WHITE,
}

class BatchGraderLogger:
    def __init__(self, log_dir='output/logs'):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'batchgrader_run_{timestamp}.log')
        self.file_logger = logging.getLogger(f'batchgrader_{timestamp}')
        self.file_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        self.file_logger.addHandler(file_handler)
        # Log a startup message immediately
        self.info(f"BatchGrader run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def _log(self, msg, level='INFO', color=None):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        color = color or LOG_COLORS.get(level, '')
        prefix = f"[{now}] {level}: "
        print(f"{color}{prefix}{msg}{Style.RESET_ALL}")
        self.file_logger.info(f'{level}: {msg}')

    def info(self, msg):
        self._log(msg, 'INFO')
    def success(self, msg):
        self._log(msg, 'SUCCESS')
    def warning(self, msg):
        self._log(msg, 'WARNING')
    def error(self, msg):
        self._log(msg, 'ERROR')
    def event(self, msg):
        self._log(msg, 'EVENT')
    def debug(self, msg):
        self._log(msg, 'DEBUG')
        
logger = BatchGraderLogger()
