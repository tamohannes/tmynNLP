from common import Registrable, Cacheable
from typing import List, Union, Any


class Tokenizer(Registrable, Cacheable):

    def __init__(self) -> None:
        pass

    def __call__(self, input: Union[str, List[str]]) -> Any:
        raise NotImplementedError

    def _atomic(self, input: str) -> Any:
        pass
