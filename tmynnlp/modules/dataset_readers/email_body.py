from typing import Dict
from cores import DatasetReader
from datasets import Dataset


@DatasetReader.register("email-body")
class EmailBodyDatasetReader(DatasetReader):

    def __init__(self, train_data_path: str, val_data_path: str, mock: bool = False) -> None:
        super().__init__(train_data_path, val_data_path, mock)

    def read(self) -> Dict[str, Dataset]:
        return super().read()

    def _read(self, path: str) -> Dataset:
        dataset_original = Dataset.from_json(path)

        body = dataset_original["body"]
        matter = dataset_original.map(
            lambda example: {'matter': example['attributes'][0]['description']})['matter']
        subject = dataset_original["subject"]

        dataset: Dataset = Dataset.from_dict(
            {'body': body, 'matter': matter, 'subject': subject})

        return dataset
