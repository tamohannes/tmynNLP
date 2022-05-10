from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
from cores import DatasetReader, Experiment, Metric, Preprocessor, Tokenizer, FeatureExtractor
from common import Device
import torch
from torch import nn


@Experiment.register("exp2")
class Exp2(Experiment):

    def __init__(self,
                 dataset_reader: DatasetReader,
                 preprocessor: Preprocessor,
                 metrics: List[Metric],
                 tokenizer: Tokenizer,
                 feature_extractor: FeatureExtractor,
                 num_workers: int = -1,
                 dist_metric: str = None,
                 batch_size: int = 4) -> None:

        super().__init__(dataset_reader, preprocessor, metrics, tokenizer,
                         feature_extractor, num_workers,
                         dist_metric=dist_metric, batch_size=batch_size)

    def __call__(self) -> Tuple[List[str], List[str]]:
        """#Base Case
        Extracting sent. embs. and label embs. and findng the best match for a sent. emb from the label embs. space
        """

        val_x = self.datasets['val']['body']
        val_y = self.datasets['val']['matter']

        y_embs = defaultdict()

        def pool(batch_out):
            # [B, S, E]
            return torch.mean(batch_out.last_hidden_state, 1)

        val_y_tok = self.tokenizer(val_y).to(Device.device)
        val_y_out = self.feature_extractor(val_y_tok)

        y_emb = val_y_out.pooler_output

        for idx, y in enumerate(val_y):
            y_embs[y] = y_emb[idx]

        def _score(dist_metric=None):
            predictions = []
            gold_labels = []

            val_x_tok = self.tokenizer(val_x).to(Device.device)
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

                y = val_y[idx]
                y_hat = max(similarities, key=similarities.get)

                predictions.append(y_hat)
                gold_labels.append(y)

            return gold_labels, predictions

        gold_labels, predictions = _score(self.dist_metric)

        return gold_labels, predictions

    def info(self) -> str:
        return "Base Case: Zero-Shot, get the embedding from pre-trained LM (bert-base-uncased, bert-base-cased, roberta-base, roberta-large) of a x and the embeddings of all ys, and match the x to a y. For matching experimented 3 ways (product, cosine similarity, !euclidean distance)."
