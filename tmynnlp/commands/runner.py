import json
import argparse
import logging
from typing import Dict, Any
from pathlib import Path
from common import Params, Cacheable, Logger, TmpHandler
from cores import Experiment
from .command import Command


@Command.register("runner")
class Runner(Command):
    """
    runner trains and validates the given experiments, and returns the specified metrics scores.
    """

    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """tmynnlp: Runner"""
        subparser = parser.add_parser(
            self.name, description=description, help="Run the epxeriments using a config JSON file"
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

        subparser.set_defaults(fire=self._runner)

        return subparser

    def _runner(self, args: argparse.Namespace):
        Cacheable.make_dir(args.cache_dir)
        Logger.make_dir(args.log_dir)
        Logger.logging_config(args.runs_file_path, args.log_dir)
        logging.info(f"execution file path: {args.runs_file_path}")
        TmpHandler.make_dir(args.tmp_dir)

        with open(args.runs_file_path, "r") as f:
            runs = json.load(f)

        for params in runs:
            self.run(Params(params))

    def run(self, params: Dict[str, Any]) -> None:
        logging.info(f"experiment: {params['type']}")
        experiment = Experiment.from_params(params)

        logging.info(f"experiment description: {experiment.info()}")

        try:
            logging.info("starting the run")
            gold_labels, predictions = experiment()

            logging.info("starting the scoring")
            scores, n_no_prediction = experiment.score(gold_labels, predictions)

            logging.info(f"number of no_predictions: {n_no_prediction}")
            logging.info(f"------------- scores: {scores} -------------")

        except Exception as e:
            logging.error("Error at %s", "division", exc_info=e)
