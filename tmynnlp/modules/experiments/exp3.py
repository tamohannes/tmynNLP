from pathlib import Path
from typing import List, Tuple
from cores import DatasetReader, Experiment, Metric, Preprocessor, Tokenizer, FeatureExtractor


@Experiment.register("exp3")
class Exp3(Experiment):

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

        def _score():
            predictions = []
            gold_labels = []

            y_hats = self.feature_extractor(
                self.datasets['val']['body'], labels=labels)
            for y_hat in y_hats:
                predictions.append(y_hat['labels'][0])
            gold_labels = self.datasets['val']['matter']

            return gold_labels, predictions

        gold_labels, predictions = _score()

        return gold_labels, predictions

    def info(self) -> str:
        return "HuggingFace Zero-Shot-Classification Pipeline: The same procedure as for Base Case."
