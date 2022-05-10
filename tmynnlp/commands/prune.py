import argparse
from .command import Command
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import os


@Command.register("prune")
class Prune(Command):
    """
    prune removes the least recently used (LRU) files and removes the rest in the specified directories (cache, tmp, logs).
    """

    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """tmynnlp: Cleanup"""
        subparser = parser.add_parser(
            self.name, description=description, help="Prune the directories (cache, tmp, logs)"
        )

        subparser.add_argument("directories", nargs="+", default=[],
                               help="directories to remove (cache, tmp, logs)")
        subparser.add_argument("--num_of_elements", type=int,
                               help='number of elements to keep per dir', default=10)

        subparser.set_defaults(func=self._prune)

        return subparser

    def _prune(self, args: argparse.Namespace):
        for directory in args.directories:
            directory_path = Path(directory)
            if directory_path.is_dir():
                directory_files: List[Dict[str, Any]] = []

                for path, _, files in os.walk(directory_path):
                    for name in files:
                        full_path: str = Path(os.path.join(path, name))
                        directory_files.append({
                            "timestamp": os.stat(full_path).st_atime,
                            "path": full_path
                        })

                directory_files_sorted = sorted(
                    directory_files, key=lambda pair: pair["timestamp"])
                directory_files_to_remove = directory_files_sorted[:len(
                    directory_files_sorted) - args.num_of_elements]

                print(
                    f"Removing: {len(directory_files_to_remove)} {directory} elements")
                for directory_file_to_remove in tqdm(directory_files_to_remove):
                    os.remove(str(directory_file_to_remove["path"]))

                    directory_file_to_remove_subdir = "/".join(
                        directory_file_to_remove["path"].parts[:-1])
                    if not os.listdir(directory_file_to_remove_subdir):
                        os.rmdir(directory_file_to_remove_subdir)
