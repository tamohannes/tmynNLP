from typing import Dict, List, Optional
from cores import DatasetReader, Preprocessor
from datasets import Dataset, load_dataset


@DatasetReader.register("ted_multi")
class TedMultiDatasetReader(DatasetReader):

    def __init__(self, train_data_path: str, valid_data_path: str,
                       preprocessor: Preprocessor, mock_samples_num: Optional[int] = None) -> None:
        super().__init__(train_data_path, valid_data_path, preprocessor, mock_samples_num)

    def read(self) -> Dict[str, Dataset]:
        return super().read()

    def _read(self, path: str) -> Dataset:
        self.labels: List[str] = []
        dataset_original = load_dataset("ted_multi", split=path)
        dataset_translations = dataset_original["translations"]

        inputs: List[str] = []
        labels: List[str] = []

        for sample in dataset_translations:
            for idx in range(len(sample["language"])):
                if not self.mock_samples_num or (self.mock_samples_num and len(inputs) < self.mock_samples_num):
                    if sample["language"][idx] not in self.labels:
                        self.labels.append(sample["language"][idx])

                    sample_input = sample["translation"][idx]
                    label = self.labels.index(sample["language"][idx])

                    sample_preprocessed = self.preprocessor({"input": sample_input, "labels": label})

                    if sample_preprocessed["input"] and sample_preprocessed["labels"] is not None:
                        inputs.append(sample_input)
                        labels.append(label)
                else:
                    break
            else:
                continue
            break

        return Dataset.from_dict({'input': inputs, 'labels': labels})
