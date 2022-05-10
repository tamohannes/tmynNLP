from pathlib import Path
from typing import List, Tuple
from cores import DatasetReader, Experiment, Metric, Preprocessor, Tokenizer, FeatureExtractor
import pickle
from tqdm import tqdm
from common import TmpHandler
import os


@Experiment.register("exp10")
class Exp10(Experiment):

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

        self.datasets['train'] = self.preprocessor(self.datasets['train'])
        self.datasets['val'] = self.preprocessor(self.datasets['val'])

        """#Base Case"""

        matters = set(self.datasets['train']['matter'])

        def extract_keys(dataset):
            matter_keys = {}

            for matter in tqdm(matters, total=len(matters)):
                bodies = dataset.filter(
                    lambda sample: sample['matter'] == matter)['body']
                tags = self.feature_extractor(bodies)

                matter_keys[matter] = tags
            return matter_keys

        if not TmpHandler.exists(self, "matter_keys_large.tmp"):
            matter_keys = extract_keys(self.datasets['train'])

            TmpHandler.store(self, matter_keys, "matter_keys_large.tmp")
        else:
            matter_keys = TmpHandler.get(self, "matter_keys_large.tmp")

        def matter_dict_to_keys(matter_keys):
            matter_set = {}
            for matter_name in matter_keys:
                matter_set[matter_name] = set()
                for matter in matter_keys[matter_name]:
                    for key in matter:
                        matter_set[matter_name].add(key['word'])
            return matter_set

        matter_set = matter_dict_to_keys(matter_keys)

        val_bodies_keys = []
        val_bodies_ners = self.feature_extractor(self.datasets['val']['body'])
        for i, body_ners in enumerate(tqdm(val_bodies_ners, total=len(val_bodies_ners))):
            keys = set()
            for tag in body_ners:
                keys.add(tag['word'])

            val_bodies_keys.append((self.datasets['val'][i]['matter'], keys))

        predictions = []
        gold_labels = []

        for sample in val_bodies_keys:
            y, keys = sample
            y_hat, overlap = "", 0

            for matter in matter_set:
                matter_keys = matter_set[matter]
                if len(matter_keys & keys) > overlap:
                    y_hat = matter
                    overlap = len(matter_keys & keys)

            predictions.append(y_hat)
            gold_labels.append(y)

        return gold_labels, predictions

    def info(self) -> str:
        return "Simplified case of 9: Instead of building graphs, use bag-of-words where words are NEs, and simply find the biggest overlap between a given x  and y's sets, by just calculating the sets' intersection."
