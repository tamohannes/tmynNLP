import logging
from typing import Dict
from common import Registrable, Cacheable
from .preprocessor import Preprocessor
from datasets import Dataset, DatasetDict


class DatasetReader(Registrable, Cacheable):

    def __init__(self, train_data_path: str, valid_data_path: str,
                       preprocessor: Preprocessor, mock_samples_num: int = None) -> None:
        self.mock_samples_num = mock_samples_num
        self.preprocessor = preprocessor
        self.data_paths: Dict[str, str] = {
            'train': train_data_path,
            'valid': valid_data_path
        }

    def read(self) -> Dict[str, Dataset]:
        datasets: DatasetDict[str, Dataset] = DatasetDict()
        for subset, path in self.data_paths.items():
            datasets[subset] = self._read(path)
            logging.info(
                f"red subset '{subset}', size: {len(datasets[subset])}")

        return datasets

    def _read(self, path: str) -> Dataset:
        raise NotImplementedError
