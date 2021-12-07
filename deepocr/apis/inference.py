import warnings

import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from ..datasets.pipelines import Compose
from ..models import build_recognizer


def init_recognizer(config, checkpoint=None, device="cuda:0"):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional): Determine which device to be used to store model.
            Default to 'cuda:0'.
    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError(
            "config must be a filename or Config object, " f"but got {type(config)}"
        )
    config.model.pretrained = None
    model = build_recognizer(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage(object):
    """A simple pipeline to load image."""

    def __init__(self, color_type="color"):
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
            color_type (str): 'Color' or 'grayscale'.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results["img"], str):
            results["filename"] = results["img"]
            results["ori_filename"] = results["img"]
        else:
            results["filename"] = None
            results["ori_filename"] = None
        img = mmcv.imread(results["img"], flag=self.color_type)
        results["img"] = img
        results["img_fields"] = ["img"]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        return results


def inference_recognizer(model, img, color_type="color"):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.
        color_type (str): 'Color' or 'grayscale'.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    inference_pipeline = [LoadImage(color_type=color_type)] + cfg.inference_pipeline[1:]
    inference_pipeline = Compose(inference_pipeline)
    # prepare data
    data = dict(img=img)
    data = inference_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        warnings.warn("We set use_torchvision=True in CPU mode.")
        # just get the actual data from DataContainer
        data["img_metas"] = data["img_metas"][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


async def async_inference_recognizer(model, img, color_type="color"):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.
        color_type (str): 'Color' or 'grayscale'.

    Returns:
        Awaitable detection results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    inference_pipeline = [LoadImage(color_type=color_type)] + cfg.inference_pipeline[1:]
    inference_pipeline = Compose(inference_pipeline)
    # prepare data
    data = dict(img=img)
    data = inference_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    result = await model.aforward_test(rescale=True, **data)
    return result
