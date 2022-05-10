from pathlib import Path
from typing import List, Tuple
from cores import DatasetReader, Experiment, Metric, Preprocessor, Tokenizer, FeatureExtractor
from collections import Counter


@Experiment.register("exp4")
class Exp4(Experiment):

    def __init__(self,
                 dataset_reader: DatasetReader,
                 preprocessor: Preprocessor,
                 metrics: List[Metric],
                 tokenizer: Tokenizer,
                 feature_extractor: FeatureExtractor,
                 num_workers: int = -1) -> None:

        super().__init__(dataset_reader, preprocessor, metrics,
                         tokenizer, feature_extractor, num_workers)

    def __call__(self) -> Tuple[List[str], List[str]]:

        self.datasets['val'] = self.preprocessor(self.datasets['val'])

        labels = list(set(self.datasets['val']['matter']))

        predictions = []
        gold_labels = []

        y_hats = self.feature_extractor(self.datasets['val']['body'], labels)
        for y_hat in y_hats:
            predictions.append(Counter(y_hat['labels']).most_common()[0][0])
        gold_labels = self.datasets['val']['matter']

        return gold_labels, predictions

    def info(self) -> str:
        return "Sub-Labeling Strategy: Instead of finding a label for a x, label every sentence x1, x2, ... xn and assign the most frequent label to x."
