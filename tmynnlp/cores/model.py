from common import Registrable
from typing import Any
import torch.nn as nn


class Model(Registrable, nn.Module):
    def forward(self) -> Any:
        raise NotImplementedError
