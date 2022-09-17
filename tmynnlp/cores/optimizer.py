from typing import Dict, Any, Generator
import torch.optim as optim


def Optimizer(model_params: Generator, args: Dict[str, Any]) -> Any:
    var = getattr(optim, args.pop("type"))
    return var(model_params, **args)
