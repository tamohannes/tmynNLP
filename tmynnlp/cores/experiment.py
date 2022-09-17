from typing import Dict, List, Any, Tuple
from .dataset_reader import DatasetReader, Dataset
from .tokenizer import Tokenizer
from .model import Model
from .tracker import Tracker
from .metric import Metric
from common import Registrable, Cacheable, Params
import multiprocessing


class Experiment(Registrable):

    def __init__(self,
                 dataset_reader: DatasetReader,
                 metrics: List[Metric],
                 tokenizer: Tokenizer = None,
                 model: Model = None,
                 tracker: Tracker = None,
                 run_params: Params = None,
                 num_workers: int = -1,
                 **kwargs) -> None:

        self.dataset_reader = dataset_reader
        self.metrics = metrics
        self.tokenizer = tokenizer
        self.model = model
        self.tracker = tracker
        self.num_workers = num_workers if num_workers > 0 else multiprocessing.cpu_count()

        self.__dict__.update(kwargs)
        self.reverse_registration()

        run_params["description"] = self.description()
        self.tracker.set_params(run_params)
        self.datasets: Dict[str, Dataset] = self.dataset_reader.read()

    def reverse_registration(self) -> None:
        properties = vars(self)

        for dependency in properties.values():
            if isinstance(dependency, Cacheable):
                dependency.reverse_registration(self)

    def __call__(self) -> Tuple[List[str], List[str]]:
        raise NotImplementedError

    def score(self, gold_labels: List[int], predictions: List[int]) -> Tuple[Dict[str, Any], int]:
        accumulated_scores: Dict[str, float] = {}

        for metric in self.metrics:
            accumulated_scores[metric.name] = metric(gold_labels, predictions)

        return accumulated_scores

    def description(self) -> str:
        raise NotImplementedError
