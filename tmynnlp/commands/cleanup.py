import argparse
from .command import Command
from pathlib import Path
import shutil


@Command.register("cleanup")
class Cleanup(Command):
    """
    cleanup removes all the specified directories (cache, tmp, logs).
    """

    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """tmynnlp: Cleanup"""
        subparser = parser.add_parser(
            self.name, description=description, help="Remove the directories (cache, tmp, logs)"
        )

        subparser.add_argument("directories", nargs="+", default=[],
                               help="directories to remove (cache, tmp, logs)")

        subparser.set_defaults(func=self._cleaner)

        return subparser

    def _cleaner(self, args: argparse.Namespace):
        for directory in args.directories:
            directory_path = Path(directory)
            if directory_path.is_dir():
                print(
                    f"Are you sure you want to delete: {directory_path} [y/N]")
                response = str(input())
                if response.lower() == 'y':
                    print(f"Removing: the {directory}")
                    shutil.rmtree(directory_path)
