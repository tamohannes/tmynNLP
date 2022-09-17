from typing import Union, Dict, List, Any
from common import Registrable, Cacheable


class Preprocessor(Registrable, Cacheable):

    def __init__(self) -> None:
        pass

    def __call__(self, examples: Union[Dict[Any, Any], Dict[List, List]]) -> Dict[Any, Union[Any, List]]:
        raise NotImplementedError
