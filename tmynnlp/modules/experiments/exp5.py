from pathlib import Path
from typing import List, Tuple
from cores import DatasetReader, Experiment, Metric, Preprocessor, Tokenizer, FeatureExtractor
from common import Device
from torch import nn
from collections import defaultdict


@Experiment.register("exp5")
class Exp5(Experiment):

    def __init__(self,
                 dataset_reader: DatasetReader,
                 preprocessor: Preprocessor,
                 metrics: List[Metric],
                 tokenizer: Tokenizer,
                 feature_extractor: FeatureExtractor,
                 ner_extractor: FeatureExtractor,
                 num_workers: int = -1,
                 dist_metric: str = None,
                 batch_size: int = 4) -> None:

        super().__init__(dataset_reader, preprocessor, metrics, tokenizer,
                         feature_extractor, num_workers,
                         ner_extractor=ner_extractor, dist_metric=dist_metric,
                         batch_size=batch_size)

    def __call__(self) -> Tuple[List[str], List[str]]:

        self.datasets['train'] = self.preprocessor(self.datasets['train'])
        self.datasets['val'] = self.preprocessor(self.datasets['val'])

        """#Base Case"""

        y_embs = defaultdict()

        val_y_tok = self.tokenizer(self.datasets['val']['matter'])
        val_y_out = self.feature_extractor(val_y_tok)

        y_emb = val_y_out.pooler_output

        for idx, y in enumerate(self.datasets['val']['matter']):
            y_embs[y] = y_emb[idx]

        def _score(dist_metric=None):
            predictions = []
            gold_labels = []

            val_x_ner = self.ner_extractor(self.datasets['val']['body'])

            val_x_ners = []
            for sample in val_x_ner:
                sample_ners = []
                for ner in sample:
                    sample_ners.append(ner['word'])
                val_x_ners.append(" ".join(sample_ners))

            val_x_tok = self.tokenizer(val_x_ners).to(Device.device)
            val_x_out = self.feature_extractor(val_x_tok)
            val_x_emb = val_x_out.pooler_output.squeeze()

            for idx, sample_emb in enumerate(val_x_emb):
                similarities = {}
                for y_emb in y_embs:
                    if not dist_metric:
                        sim = y_embs[y_emb].T @ sample_emb
                    elif dist_metric == 'cosine':
                        cos = nn.CosineSimilarity(dim=0)
                        sim = cos(y_embs[y_emb], sample_emb)
                    elif dist_metric == 'euclidean':
                        pdist = nn.PairwiseDistance(p=2)
                        sim = 800 - pdist(y_embs[y_emb], sample_emb)
                    similarities[y_emb] = sim

                y = self.datasets['val']['matter'][idx]
                y_hat = max(similarities, key=similarities.get)

                predictions.append(y_hat)
                gold_labels.append(y)

            return gold_labels, predictions

        gold_labels, predictions = _score(self.dist_metric)

        return gold_labels, predictions

    def info(self) -> str:
        return "Similar to 2, but instead of getting the embs of the entire sentence get the Named Entities."
