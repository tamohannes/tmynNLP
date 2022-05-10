import logging
from typing import Dict
from common import Registrable
from datasets import Dataset, DatasetDict


class DatasetReader(Registrable):

    def __init__(self, train_data_path: str, val_data_path: str, mock: bool = False) -> None:
        self.mock = mock
        self._mock_samples_num = 10
        self.data_paths: Dict[str, str] = {
            'train': train_data_path,
            'val': val_data_path
        }

    def read(self) -> Dict[str, Dataset]:
        datasets: DatasetDict[str, Dataset] = DatasetDict()
        for subset, path in self.data_paths.items():
            if self.mock:
                datasets[subset] = Dataset.from_dict(
                    self._read(path)[:self._mock_samples_num])
            else:
                datasets[subset] = self._read(path)
            logging.info(f"red subset '{subset}', size: {len(datasets[subset])}")

        return datasets

    def _read(self, path: str) -> Dataset:
        raise NotImplementedError
