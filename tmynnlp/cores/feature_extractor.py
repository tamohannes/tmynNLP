from common import Registrable, Cacheable
from typing import Any


class FeatureExtractor(Registrable, Cacheable):

    def __init__(self) -> None:
        raise NotImplementedError

    def __call__(self, input: Any, **kwargs) -> Any:
        raise NotImplementedError
