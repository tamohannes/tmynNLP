from cores import Metric
from sklearn.metrics import f1_score
import torch


@Metric.register("f1")
class F1(Metric):
    name = "f1"

    def __init__(self, average: str = "weighted") -> None:
        super().__init__()
        self.average = average

    def get_metric(self, reset: bool = False) -> float:
        metric: float = f1_score(torch.cat(self._gold_labels), torch.cat(self._predictions), average=self.average)
        if reset:
            self._reset()
        return metric
