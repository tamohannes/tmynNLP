from pathlib import Path
from typing import Dict, List, Tuple
from cores import DatasetReader, Experiment, Metric, Preprocessor, Tokenizer, FeatureExtractor
from common import TmpHandler
from collections import defaultdict
import torch
from torch import nn
from tqdm import tqdm


@Experiment.register("exp8")
class Exp8(Experiment):

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

        self.datasets['train'] = self.preprocessor(self.datasets['train'])
        self.datasets['val'] = self.preprocessor(self.datasets['val'])

        """#Base Case"""
        if not TmpHandler.exists(self, "matter_reps.tmp"):
            matter_reps: Dict[str, List[torch.tensor]] = defaultdict(list)

            batch_toks = self.tokenizer(self.datasets['train']['body'])
            batch_outs = self.feature_extractor(batch_toks)

            embs = batch_outs.pooler_output.detach().cpu()

            for idx, matter in enumerate(self.datasets['train']['matter']):
                matter_reps[matter].append(embs[idx])

            TmpHandler.store(self, matter_reps, "matter_reps.tmp")
        else:
            matter_reps: Dict[str, List[torch.tensor]] = TmpHandler.get(self, "matter_reps.tmp")

        # pool
        matter_reps_pooled: Dict[str, List[torch.tensor]
                                 ] = defaultdict(torch.tensor)
        for matter in matter_reps:
            n = len(matter_reps[matter])
            m = matter_reps[matter][0].shape[0]
            vecs = matter_reps[matter]

            rep = torch.zeros(n, m)
            for i in range(rep.shape[0]):
                rep[i] = vecs[i]

            rep_mean = rep.mean(axis=0)
            matter_reps_pooled[matter] = rep_mean

        y_embs = defaultdict()

        val_y_batch_toks = self.tokenizer(self.datasets['val']['body'])
        val_y_batch_outs = self.feature_extractor(val_y_batch_toks)
        y_emb = val_y_batch_outs.pooler_output.detach().cpu()

        for idx, y in enumerate(self.datasets['val']['body']):
            y_embs[y] = y_emb[idx]

        def _score(dist_metric):
            val_y = self.datasets['val']['matter']
            predictions = []
            gold_labels = []

            for i, y_body in tqdm(enumerate(y_embs), total=len(y_embs)):
                similarities = {}
                y = val_y[i]

                for matter_rep in matter_reps_pooled:
                    sample_emb = matter_reps_pooled[matter_rep]

                    if not dist_metric:
                        sim = y_embs[y_body].T @ sample_emb
                    elif dist_metric == 'cosine':
                        cos = nn.CosineSimilarity(dim=0)
                        sim = cos(y_embs[y_body], sample_emb)
                    elif dist_metric == 'euclidean':
                        pdist = nn.PairwiseDistance(p=2)
                        sim = 800 - pdist(y_embs[y_body], sample_emb)
                    else:
                        return
                    similarities[matter_rep] = sim

                y_hat = max(similarities, key=similarities.get)

                predictions.append(y_hat)
                gold_labels.append(y)

            return gold_labels, predictions

        gold_labels, predictions = _score(self.dist_metric)

        return gold_labels, predictions

    def info(self) -> str:
        return "Similar to 2. Matter Representation Change: Instead of getting the LM given embedding of the matter, construct the matter representation via the prior knowledge (using the previous documents information - averaging the embeddings of the documents of a matter)."
