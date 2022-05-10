from common import Registrable
from typing import List, Any, Tuple


class Metric(Registrable):

    def __init__(self) -> None:
        pass

    def __call__(self, gold_labels: List[str], predictions: List[str]) -> Any:
        raise NotImplementedError

    def extract_no_predictions(self, gold_labels: List[str], predictions: List[str]) -> Tuple[List[str], List[str], int]:
        no_predictions: List[int] = []

        for i, (gold_label, predicton) in enumerate(zip(gold_labels, predictions)):
            if predicton == "":
                no_predictions.append(i)

        popped_gold_labels: List[str] = []
        popped_predictions: List[str] = []
        for i in range(len(gold_labels)):
            if i not in no_predictions:
                popped_gold_labels.append(gold_labels[i])
                popped_predictions.append(predictions[i])

        return popped_gold_labels, popped_predictions, len(no_predictions)
