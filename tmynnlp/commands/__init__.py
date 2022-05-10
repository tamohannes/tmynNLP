from .runner import Runner
from .cleanup import Cleanup
from .prune import Prune

from .command import Command
from typing import Tuple
import argparse
from common.util import import_module_and_submodules


def parse_args() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    """
    Creates the argument parser for the main program and uses it to parse the args.
    """
    parser = argparse.ArgumentParser(description='Run tmynnlp')

    subparsers = parser.add_subparsers(title="Commands", metavar="")

    def add_subcommands():
        for subcommand_name in sorted(Command.list_available()):
            subcommand_class = Command.by_name(subcommand_name)
            subcommand = subcommand_class()
            subcommand.add_subparser(subparsers)

    # Add all default registered subcommands first.
    add_subcommands()
    # Now we can parse the arguments.
    args = parser.parse_args()

    return parser, args


def main():
    parser, args = parse_args()

    # If a subparser is triggered, it adds its work(it's main executon method) as `args.fire`.
    # So if no such attribute has been added, no subparser was triggered, so give the user some help.
    if "fire" in dir(args):
        # Import any additional modules needed (to register custom classes).
        for package_name in getattr(args, "include_package", []):
            import_module_and_submodules(package_name)

        args.fire(args)
    else:
        parser.print_help()
