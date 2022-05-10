from common import Registrable, Cacheable
from datasets import Dataset


class Preprocessor(Registrable, Cacheable):

    def __init__(self) -> None:
        pass

    def __call__(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError
