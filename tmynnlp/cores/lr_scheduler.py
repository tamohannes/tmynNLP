from typing import Dict, Any
import torch.optim.lr_scheduler as lr_scheduler


def LRScheduler(optimizer, args: Dict[str, Any]) -> Any:
    var = getattr(lr_scheduler, args.pop("type"))
    return var(optimizer, **args)
