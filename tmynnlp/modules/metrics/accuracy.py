from typing import List, Tuple
from cores import Metric
from sklearn.metrics import accuracy_score


@Metric.register("accuracy")
class Accuracy(Metric):
    name = "accuracy"

    def __init__(self) -> None:
        pass

    def __call__(self, gold_labels: List[str], predictions: List[str]) -> Tuple[float, int]:
        gold_labels, predictions, n_no_prediction = self.extract_no_predictions(gold_labels, predictions)
        return accuracy_score(gold_labels, predictions), n_no_prediction
