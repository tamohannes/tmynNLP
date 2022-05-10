from typing import List
from cores import Metric
from sklearn.metrics import f1_score


@Metric.register("f1")
class F1(Metric):
    name = "f1"

    def __init__(self, average: str = "weighted") -> None:
        self.average = average

    def __call__(self, gold_labels: List[str], predictions: List[str]) -> Tuple[float, int]:
        gold_labels, predictions, n_no_prediction = self.extract_no_predictions(gold_labels, predictions)
        return f1_score(gold_labels, predictions, average=self.average), n_no_prediction
