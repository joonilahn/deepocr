"""This code was copied from the mmcv library."""


import torch.nn as nn

from mmcv.utils import Registry, build_from_cfg

from .losses import register_torch_losses

BACKBONES = Registry("backbone")
PRETRANSFORMS = Registry("pretransform")
RECOGNIZERS = Registry("recognizer")
ENCODERS = Registry("encoder")
DECODERS = Registry("decoder")
POSENCODERS = Registry("posencoder")
LOSSES = Registry("loss")
LOSSES = register_torch_losses(LOSSES)
NECKS = Registry("neck")


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_recognizer(cfg):
    """Build recognizer."""
    return build(cfg, RECOGNIZERS)


def build_pretransform(cfg):
    """Build pretransform."""
    return build(cfg, PRETRANSFORMS)


def build_encoder(cfg):
    """Build encoder."""
    return build(cfg, ENCODERS)


def build_decoder(cfg):
    """Build decoder."""
    return build(cfg, DECODERS)


def build_posencoder(cfg):
    """Build positional encoder."""
    return build(cfg, POSENCODERS)


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)
