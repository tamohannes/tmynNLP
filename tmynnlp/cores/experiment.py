from pathlib import Path
from typing import Dict, List, Any, Tuple
from .feature_extractor import FeatureExtractor
from .preprocessor import Preprocessor
from .tokenizer import Tokenizer
from .metric import Metric
from .dataset_reader import DatasetReader, Dataset
from common import Registrable, Cacheable
import multiprocessing


class Experiment(Registrable):

    def __init__(self,
                 dataset_reader: DatasetReader,
                 preprocessor: Preprocessor,
                 metrics: List[Metric],
                 tokenizer: Tokenizer = None,
                 feature_extractor: FeatureExtractor = None,
                 num_workers: int = -1,
                 **kwargs) -> None:

        self.dataset_reader = dataset_reader
        self.datasets: Dict[str, Dataset] = self.dataset_reader.read()
        self.preprocessor = preprocessor
        self.metrics = metrics
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.num_workers = num_workers if num_workers > 0 else multiprocessing.cpu_count()

        self.__dict__.update(kwargs)

        self.reverse_registration()

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
            accumulated_scores[metric.name], n_no_prediction = metric(gold_labels, predictions)

        return accumulated_scores, n_no_prediction

    def info(self) -> str:
        raise NotImplementedError
