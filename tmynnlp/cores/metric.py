from common import Registrable
from typing import List, Any


class Metric(Registrable):

    def __init__(self) -> None:
        self._gold_labels: List[Any] = []
        self._predictions: List[Any] = []

    def __call__(self, gold_labels: List[Any], predictions: List[Any]) -> None:
        self._gold_labels.append(gold_labels.cpu())
        self._predictions.append(predictions.cpu())

    def _reset(self):
        self._gold_labels: List[Any] = []
        self._predictions: List[Any] = []

    def get_metric(self, reset: bool = False):
        raise NotImplementedError
