from typing import Dict, Any
import torch.nn as nn


def Criterion(args: Dict[str, Any]) -> Any:
    var = getattr(nn, args["type"])
    return var()
