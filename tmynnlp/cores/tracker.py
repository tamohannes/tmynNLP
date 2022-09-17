from typing import Dict, Any
from common import Registrable, Cacheable


class Tracker(Registrable, Cacheable):

    def __init__(self) -> None:
        pass

    def set_params(self, args: Dict[str, Any]):
        pass

    def track(self, logs: Dict[str, Any] = None):
        pass
