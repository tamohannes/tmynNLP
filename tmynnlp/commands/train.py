import json
import argparse
import logging
import copy
from common import Params, Cacheable, Logger, TmpHandler
from cores import Experiment
from .command import Command


@Command.register("train")
class Train(Command):
    """
    Trains and validates the given experiments, and returns the specified metrics scores.
    """

    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """tmynnlp: Train"""
        subparser = parser.add_parser(
            self.name, description=description, help="Train the epxeriments using a config JSON file"
        )

        subparser.add_argument('runs_file_path', type=str,
                               help='runs file path(json)')
        subparser.add_argument("--include_package", type=str,
                               action="append", default=[],
                               help="additional packages to include")
        subparser.add_argument('--log_dir', type=str,
                               help='logging directory', default="./logs")
        subparser.add_argument('--tmp_dir', type=str,
                               help='tmp files directory', default="./tmp")
        subparser.add_argument('--cache_dir', type=str,
                               help='cache files directory', default="./cache")

        subparser.set_defaults(fire=self._train)

        return subparser

    def _train(self, args: argparse.Namespace):
        Cacheable.make_dir(args.cache_dir)
        Logger.make_dir(args.log_dir)
        Logger.logging_config(args.runs_file_path, args.log_dir)
        logging.info(f"execution file path: {args.runs_file_path}")
        TmpHandler.make_dir(args.tmp_dir)

        with open(args.runs_file_path, "r") as f:
            runs = json.load(f)

        for params in runs:
            self.train(Params(params))

    def train(self, params: Params) -> None:
        logging.info(f"experiment: {params['type']}")
        experiment = Experiment.from_params(
            params, run_params=copy.deepcopy(params.params))

        logging.info(f"experiment description: {experiment.description()}")

        try:
            logging.info("starting the training")
            experiment()

        except Exception as e:
            logging.error("Error at %s", "division", exc_info=e)
