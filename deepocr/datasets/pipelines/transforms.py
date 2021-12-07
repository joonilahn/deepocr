"""
openmmlab's mmdetection transforms and custom pipelines.
Many lines in this code were copied and modified from openmmlab's mmdetection 2.2.1 transformss module.
https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/pipelines/transforms.py
"""

import inspect

import cv2
import mmcv
import numpy as np
import torch
from numpy import random

from ..builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@PIPELINES.register_module()
class Resize(object):
    """Resize images.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio range
      and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(
        self, img_scale=None, multiscale_mode="range", ratio_range=None, keep_ratio=True
    ):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ["value", "range"]

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range
            )
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == "range":
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == "value":
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results["scale"] = scale
        results["scale_idx"] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get("img_fields", ["img"]):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key], results["scale"], return_scale=True
                )
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key], results["scale"], return_scale=True
                )
            results[key] = img

            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32
            )
            results["img_shape"] = img.shape
            # in case that there is no padding
            results["pad_shape"] = img.shape
            results["scale_factor"] = scale_factor
            results["keep_ratio"] = self.keep_ratio

    def __call__(self, results):
        """Call function to resize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if "scale" not in results:
            if "scale_factor" in results:
                img_shape = results["img"].shape[:2]
                scale_factor = results["scale_factor"]
                assert isinstance(scale_factor, float)
                results["scale"] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1]
                )
            else:
                self._random_scale(results)
        else:
            assert (
                "scale_factor" not in results
            ), "scale and scale_factor cannot be both set."

        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(img_scale={self.img_scale}, "
        repr_str += f"multiscale_mode={self.multiscale_mode}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"keep_ratio={self.keep_ratio})"
        return repr_str


@PIPELINES.register_module()
class RotateIfNeeded(object):
    """Rotate image if height of the image is longer than the width.

    Args
        angle (float): Rotation angle. by default -90.
    """

    def __init__(self, angle=-90):
        super(RotateIfNeeded, self).__init__()
        self.angle = angle

    def _rotate_img(self, results):
        img = results["img"]
        h, w = img.shape[:2]
        img_ratio = h / w
        if img_ratio > 1.0:
            results["img"] = mmcv.imrotate(img, self.angle, auto_bound=True)

    def __call__(self, results):
        """Call function to rotate images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """
        self._rotate_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(angle={self.angle}, "


@PIPELINES.register_module()
class MaxHeightResize(object):
    """Rescale an image so that height is equal to max_height, keeping the aspect ratio of the initial image.

    Args:
        max_height (int): maximum height of the image after the transformation.
        max_width (int): maximum width of the image after the transformation.

    Image types:
        uint8, float32
    """

    def __init__(self, max_height, max_width):
        super(MaxHeightResize, self).__init__()
        self.max_height = max_height
        self.max_width = max_width

    def _calculate_scale(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            h, w = img.shape[:2]
            h_scale = self.max_height / h
            w_rescale = int(w * h_scale)
            if w_rescale > self.max_width:
                w_rescale = self.max_width
                h_rescale = int(h * self.max_width / w)
            else:
                h_rescale = int(self.max_height)
            results["scale"] = (w_rescale, h_rescale)

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get("img_fields", ["img"]):
            img, w_scale, h_scale = mmcv.imresize(
                results[key], results["scale"], return_scale=True
            )
            results[key] = img
            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32
            )
            results["img_shape"] = img.shape
            # in case that there is no padding
            results["pad_shape"] = img.shape
            results["scale_factor"] = scale_factor
            results["keep_ratio"] = True

    def __call__(self, results):
        """Call function to resize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """
        self._calculate_scale(results)
        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"max_height={self.max_height}, "
        return repr_str


@PIPELINES.register_module()
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(
            self,
            size=None,
            size_divisor=None,
            pad_val=0,
            padding_side="even",
            padding_mode="constant",
        ):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.padding_side = padding_side
        self.padding_mode = padding_mode
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _get_padding_points(self, pad_size, img_size, padding_side):
        # get padding points for top and bottom
        if pad_size[0] <= img_size[0]:
            top = 0
            bottom = 0
        else:
            pad_height = pad_size[0] - img_size[0]
            if padding_side == "random":
                top = random.randint(0, pad_height)
            else:
                top = pad_height // 2
            bottom = pad_height - top

        # get padding points for left and right
        if pad_size[1] <= img_size[1]:
            left = 0
            right = 0
        else:
            pad_width = pad_size[1] - img_size[1]
            if padding_side == "random":
                left = random.randint(0, pad_width)
                right = pad_width - left
            elif  padding_side == "even":
                left = pad_width // 2
                right = pad_width - left
            elif padding_side == "right":
                left = 0
                right = pad_width
            elif padding_side == "left":
                left = pad_width
                right = 0
            else:
                raise ValueError

        return (top, bottom, left, right)

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for key in results.get("img_fields", ["img"]):
            if self.size is not None:
                img = results[key]
                img_shape = results[key].shape
                pad_size = self.size
                if len(pad_size) < len(img_shape):
                    pad_size = pad_size + (img_shape[-1],)
                assert len(pad_size) == len(img_shape)
                for s, img_s in zip(pad_size, img_shape):
                    assert (
                        s >= img_s
                    ), "Resized image size {} should be less than or equal to pad side {}.".format(
                        img_s, s
                    )
                top, bottom, left, right = self._get_padding_points(
                    pad_size, img_shape, self.padding_side
                )
                if self.padding_mode == "constant":
                    padded_img = cv2.copyMakeBorder(
                        img, top, bottom, left, right, cv2.BORDER_CONSTANT, 0
                    )
                elif self.padding_mode == "replicate":
                    padded_img = cv2.copyMakeBorder(
                        img, top, bottom, left, right, cv2.BORDER_REPLICATE
                    )
                else:
                    raise ValueError

            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=self.pad_val
                )
            results[key] = padded_img
        results["pad_shape"] = padded_img.shape
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get("img_fields", ["img"]):
            results[key] = mmcv.imnormalize(
                results[key], self.mean, self.std, self.to_rgb
            )
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        if "img_fields" in results:
            assert results["img_fields"] == ["img"], "Only single img_fields is allowed"
        img = results["img"]
        assert img.dtype == np.float32, (
            "PhotoMetricDistortion needs the input image of dtype np.float32,"
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        )
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower, self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        results["img"] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@PIPELINES.register_module()
class Corrupt(object):
    """Corruption augmentation.

    Corruption transforms implemented based on
    `imagecorruptions <https://github.com/bethgelab/imagecorruptions>`_.

    Args:
        corruption (str): Corruption name.
        severity (int, optional): The severity of corruption. Default: 1.
    """

    def __init__(self, corruption, severity=1, p=1.0):
        self.corruption = corruption
        self.severity = severity
        self.p = p

    def __call__(self, results):
        """Call function to corrupt image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images corrupted.
        """

        if corrupt is None:
            raise RuntimeError("imagecorruptions is not installed")
        if "img_fields" in results:
            assert results["img_fields"] == ["img"], "Only single img_fields is allowed"
        if np.random.rand() < self.p:
            results["img"] = corrupt(
                results["img"].astype(np.uint8),
                corruption_name=self.corruption,
                severity=self.severity,
            )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(corruption={self.corruption}, "
        repr_str += f"severity={self.severity}, "
        repr_str += f"p={self.p})"
        return repr_str


@PIPELINES.register_module()
class RandomRotate(object):
    """Randomly rotate the images.

    Args:
        limit (tuple or list or int or float: limit of the rotation angle. angle will be selected from uniform distribution.
        border_mode (str): cv2 border_mode. "reflect" or "constant". Default: "reflect".
        p (float or int): probability. Default: 1.0.
    """

    def __init__(self, limit, border_mode="reflect", auto_bound=False, p=1.0):
        if isinstance(limit, list) or isinstance(limit, tuple):
            assert len(limit) == 2
            self.limit = limit
        else:
            assert isinstance(limit, float) or isinstance(limit, int)
            self.limit = (-np.abs(limit), np.abs(limit))
        if border_mode == "constant":
            self.border_mode = cv2.BORDER_CONSTANT
        elif border_mode == "reflect":
            self.border_mode = cv2.BORDER_REFLECT_101
        elif border_mode == "replicate":
            self.border_mode = cv2.BORDER_REPLICATE
        else:
            raise ValueError
        self.auto_bound = auto_bound
        self.p = p

    def imrotate(
        self,
        img,
        angle,
        border_mode,
        center=None,
        scale=1.0,
        border_value=0,
        auto_bound=False,
    ):
        """Rotate an image.

        Args:
            img (ndarray): Image to be rotated.
            angle (float): Rotation angle in degrees, positive values mean
                clockwise rotation.
            center (tuple[float], optional): Center point (w, h) of the rotation in
                the source image. If not specified, the center of the image will be
                used.
            scale (float): Isotropic scale factor.
            border_value (int): Border value.
            auto_bound (bool): Whether to adjust the image size to cover the whole
                rotated image.

        Returns:
            ndarray: The rotated image.
        """
        if center is not None and auto_bound:
            raise ValueError("`auto_bound` conflicts with `center`")
        h, w = img.shape[:2]
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        assert isinstance(center, tuple)

        matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        if auto_bound:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = h * sin + w * cos
            new_h = h * cos + w * sin
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))
        rotated = cv2.warpAffine(
            img, matrix, (w, h), borderMode=border_mode, borderValue=border_value
        )
        return rotated

    def __call__(self, results):
        """Call function to corrupt image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images corrupted.
        """
        if "img_fields" in results:
            assert results["img_fields"] == ["img"], "Only single img_fields is allowed"
        if np.random.rand() < self.p:
            img = results["img"]
            angle = random.uniform(self.limit[0], self.limit[1])
            results["img"] = self.imrotate(
                img, angle, self.border_mode, auto_bound=self.auto_bound
            )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(limit={self.limit}, "
        repr_str += f"(border_mode={self.border_mode}, "
        repr_str += f"(auto_bound={self.auto_bound}, "
        repr_str += f"p={self.p})"
        return repr_str


@PIPELINES.register_module()
class OneOfCorrupt(object):
    """Select one of Corruption augmentation.
    Args:
        corruptions (list[dict]): List of corruption names.
        p (float): Probaility of applying the corruption. Default: 1.
    """

    def __init__(self, corruptions, p=1.0):
        assert isinstance(corruptions, list)
        self.corruptions = corruptions
        self.num_corruptions = len(self.corruptions)
        self.p = p

    def __call__(self, results):
        """Call function to corrupt image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images corrupted.
        """
        img = results["img"]

        if corrupt is None:
            raise RuntimeError("imagecorruptions is not installed")
        if "img_fields" in results:
            assert results["img_fields"] == ["img"], "Only single img_fields is allowed"
        if (img.shape[0] < 32) or (img.shape[1] < 32):
            return results
        if np.random.rand() < self.p:
            corrupt_idx = np.random.randint(0, self.num_corruptions)
            corruption_dict = self.corruptions[corrupt_idx]
            corruption_name = corruption_dict["name"]
            severity_min = corruption_dict["min"]
            severity_max = corruption_dict["max"]
            severity = np.random.randint(severity_min, severity_max + 1)
            img = corrupt(
                img.astype(np.uint8),
                corruption_name=corruption_name,
                severity=severity,
            )
            if len(img.shape) != len(results["img_shape"]):
                if len(results["img_shape"]) == 2:
                    img = mmcv.bgr2gray(img)
                elif len(results["img_shape"]) == 3:
                    img = mmcv.gray2bgr(img)
            results["img"] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(corruption={self.corruptions}, "
        repr_str += f"p={self.p})"
        return repr_str


@PIPELINES.register_module()
class Albu(object):
    """Albumentation augmentation.

    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.

    An example of ``transforms`` is as followed:

    .. code-block::

        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]

    Args:
        transforms (list[dict]): A list of albu transformations
        bbox_params (dict): Bbox_params for albumentation `Compose`
        keymap (dict): Contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): Whether to skip the image if no ann left
            after aug
    """

    def __init__(
        self,
        transforms,
        bbox_params=None,
        keymap=None,
        update_pad_shape=False,
        skip_img_without_anno=False,
    ):
        if Compose is None:
            raise RuntimeError("albumentations is not installed")

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (
            isinstance(bbox_params, dict)
            and "label_fields" in bbox_params
            and "filter_lost_elements" in bbox_params
        ):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params["label_fields"]
            bbox_params["label_fields"] = ["idx_mapper"]
            del bbox_params["filter_lost_elements"]

        self.bbox_params = self.albu_builder(bbox_params) if bbox_params else None
        self.aug = Compose(
            [self.albu_builder(t) for t in self.transforms],
            bbox_params=self.bbox_params,
        )

        if not keymap:
            self.keymap_to_albu = {
                "img": "image",
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()

        obj_type = args.pop("type")
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError("albumentations is not installed")
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f"type must be a str or valid type, but got {type(obj_type)}"
            )

        if "transforms" in args:
            args["transforms"] = [
                self.albu_builder(transform) for transform in args["transforms"]
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper. Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)
        results = self.aug(**results)

        if "gt_label" in results:
            if isinstance(results["gt_label"], list):
                results["gt_label"] = np.array(results["gt_label"])
            results["gt_label"] = results["gt_label"].astype(np.int64)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results["pad_shape"] = results["img"].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f"(transforms={self.transforms})"
        return repr_str


@PIPELINES.register_module()
class InvertColor(object):
    """Invert color of the image.

    black (0) -> white (255)
    white (255) -> black (0)

    Args:
        zero_to_one_scale (bool): If true, convert values from [0,255] scale to [0,1] scale.
    """

    def __init__(self, invert_color=True, zero_to_one_scale=False):
        self.invert_color = invert_color
        self.zero_to_one_scale = zero_to_one_scale

    def invert_imgs(self, imgs):
        if self.invert_color:
            imgs = -1.0 * imgs + 255.0
        if self.zero_to_one_scale:
            imgs = imgs / 255.0
        return imgs

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            results[key] = self.invert_imgs(results[key])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(invert_color={self.invert_color},"
        repr_str += f"(zero_to_one_scale={self.zero_to_one_scale},"
        return repr_str


@PIPELINES.register_module()
class RandomSideShade(object):
    """
    Add Random shade, contrast adjusted image, to the random side of the image.

    Args:
        contrast_lower_limit (float): The limit for the contrast ratio. The lower the darker
        shade_range (tuple or list): The range of the shade. The values are between 0 and 1.
        p (float): probability
    """

    def __init__(self, contrast_lower_limit=0.1, shade_range=(0.1, 0.5), p=0.5):
        assert 0.0 < contrast_lower_limit < 0.5
        self.contrast_lower_limit = contrast_lower_limit
        self.shade_range = shade_range
        self.p = p
        self.max_values_by_type = {
            np.dtype("uint8"): 255,
            np.dtype("uint16"): 65535,
            np.dtype("uint32"): 4294967295,
            np.dtype("float32"): 1.0,
        }

    def _brightness_contrast_adjust_non_uint(
            self, img, alpha=1, beta=0, beta_by_max=False
    ):
        dtype = img.dtype
        img = img.astype("float32")

        if alpha != 1:
            img *= alpha
        if beta != 0:
            if beta_by_max:
                max_value = self.max_values_by_type[dtype]
                img += beta * max_value
            else:
                img += beta * np.mean(img)
        return img

    def _brightness_contrast_adjust_uint(self, img, alpha=1, beta=0, beta_by_max=False):
        dtype = np.dtype("uint8")

        max_value = self.max_values_by_type[dtype]

        lut = np.arange(0, max_value + 1).astype("float32")

        if alpha != 1:
            lut *= alpha
        if beta != 0:
            if beta_by_max:
                lut += beta * max_value
            else:
                lut += beta * np.mean(img)

        lut = np.clip(lut, 0, max_value).astype(dtype)
        img = cv2.LUT(img, lut)
        return img

    def _brightness_contrast_adjust(self, img, alpha=1, beta=0, beta_by_max=False):
        if img.dtype == np.uint8:
            return self._brightness_contrast_adjust_uint(img, alpha, beta, beta_by_max)
        return self._brightness_contrast_adjust_non_uint(img, alpha, beta, beta_by_max)

    def _add_random_shade(self, img):
        # get shade image (contrast adjusted image)
        contrast = np.random.uniform(self.contrast_lower_limit, 0.5)
        img_shade = self._brightness_contrast_adjust(img, alpha=contrast)

        # add the shade image to the original image
        random_side = np.random.randint(low=0, high=4)

        # 0: upper side, 1: right side, 2: botton side, 3: left side
        imgaug = img.copy()
        height, width = img.shape[0], img.shape[1]

        if random_side in (0, 2):
            shade_size = int(height * np.random.uniform(*self.shade_range))
        else:
            shade_size = int(width * np.random.uniform(*self.shade_range))

        if random_side == 0:
            imgaug[:shade_size] = img_shade[:shade_size]
        elif random_side == 1:
            imgaug[:, :shade_size] = img_shade[:, :shade_size]
        elif random_side == 2:
            imgaug[height - shade_size :] = img_shade[height - shade_size :]
        else:
            imgaug[:, width - shade_size :] = img_shade[:, width - shade_size :]

        return imgaug

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            if np.random.rand() < self.p:
                results[key] = self._add_random_shade(results[key])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"contrast_lower_limit={self.contrast_lower_limit}"
        repr_str += f"shade_range={self.shade_range}"
        repr_str += f"p={self.p}"
        return repr_str


@PIPELINES.register_module()
class RandomShade(RandomSideShade):
    """
    Add Random shade, contrast adjusted image, to the random side of the image.

    Args:
        contrast_lower_limit (float): The limit for the contrast ratio. The lower the darker
        shade_range (tuple or list): The range of the shade. The values are between 0 and 1.
        p (float): probability
    """

    def __init__(self, contrast_lower_limit=0.1, shade_range=(0.1, 0.5), p=0.5):
        super(RandomShade, self).__init__(
            contrast_lower_limit=contrast_lower_limit,
            shade_range=shade_range,
            p=p
        )

    def _add_random_shade(self, img):
        # get shade image (contrast adjusted image)
        contrast = np.random.uniform(self.contrast_lower_limit, 0.5)
        img_shade = self._brightness_contrast_adjust(img, alpha=contrast)

        # add the shade image to the original image
        random_side = np.random.randint(low=0, high=4)

        # 0: upper side, 1: right side, 2: botton side, 3: left side
        imgaug = img.copy()
        height, width = img.shape[0], img.shape[1]

        if random_side == 0:
            shade_size = int(height * np.random.uniform(*self.shade_range))
            shade_start = np.random.randint(low=0, high=height - shade_size)
            shade_end = np.clip(shade_start + shade_size, 0, height)
            imgaug[shade_start:shade_end, :] = img_shade[shade_start:shade_end, :]

        else:
            shade_size = int(width * np.random.uniform(*self.shade_range))
            shade_start = np.random.randint(low=0, high=width - shade_size)
            shade_end = np.clip(shade_start + shade_size, 0, width)
            imgaug[:, shade_start:shade_end] = img_shade[:, shade_start:shade_end]

        return imgaug

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            if np.random.rand() < self.p:
                results[key] = self._add_random_shade(results[key])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"contrast_lower_limit={self.contrast_lower_limit}"
        repr_str += f"shade_range={self.shade_range}"
        repr_str += f"p={self.p}"
        return repr_str
