import inspect

import torch


def register_torch_losses(losses):
    """Register all losses from official torch library."""
    for module_name in dir(torch.nn.modules.loss):
        if module_name.startswith("_"):
            continue
        _criterion = getattr(torch.nn.modules.loss, module_name)
        if inspect.isclass(_criterion):
            losses.register_module()(_criterion)
    return losses
