from cores import Metric
from sklearn.metrics import accuracy_score
import torch


@Metric.register("accuracy")
class Accuracy(Metric):
    name = "accuracy"

    def __init__(self) -> None:
        super().__init__()

    def get_metric(self, reset: bool = False) -> float:
        metric: float = accuracy_score(torch.cat(self._gold_labels), torch.cat(self._predictions))
        if reset:
            self._reset()
        return metric
