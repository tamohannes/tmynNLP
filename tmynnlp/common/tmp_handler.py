from pathlib import Path
from common import util
from typing import Any
import os
import pickle


class TmpHandler:
    tmp_dir: Path

    @classmethod
    def make_dir(cls, tmp_dir: str):
        TmpHandler.tmp_dir = Path(tmp_dir)
        TmpHandler.tmp_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_path(cls, instance, *args) -> Path:
        method_path = Path(util.hash(str(instance).split(" ")[0][1:]))
        dir_path = TmpHandler.tmp_dir.joinpath(method_path)
        args_str = "/".join(args)
        tmp_path = dir_path.joinpath(args_str)

        return tmp_path

    @classmethod
    def exists(cls, instance, *args) -> bool:
        tmp_file_path: Path = TmpHandler.get_path(instance, *args)
        return tmp_file_path.is_file()

    @classmethod
    def store(cls, instance, tmp_file: Any, *args) -> None:
        tmp_file_path: Path = TmpHandler.get_path(instance, *args)

        os.makedirs(os.path.dirname(tmp_file_path), exist_ok=True)
        with open(tmp_file_path, "wb") as f:
            pickle.dump(tmp_file, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def get(cls, instance, *args) -> Any:
        tmp_file_path: Path = TmpHandler.get_path(instance, *args)

        with open(tmp_file_path, "rb") as f:
            tmp_file = pickle.load(f)

        return tmp_file
