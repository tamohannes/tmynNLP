import os
import datetime
import logging
import logging.config
from pathlib import Path


class Logger():
    log_dir: Path

    @classmethod
    def make_dir(cls, log_dir: str):
        Logger.log_dir = Path(log_dir)
        Logger.log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def logging_config(self, execution_file_path: str, log_dir: str) -> None:
        execution_file_path: Path = Path(execution_file_path)
        file_name: str = execution_file_path.stem

        logging.basicConfig(
            filename=os.path.join(
                log_dir, f"{'{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())}_{file_name}.log"),
            level=logging.DEBUG,
            format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s->%(funcName)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
