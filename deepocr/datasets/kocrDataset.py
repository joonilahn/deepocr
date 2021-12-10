"""
Many lines in this code were copied and modified from the openmmlab's mmdetection 2.2.1 Datasets module.
https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/
"""

import os
import random
import re

import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS, build_converter
from .eval_utils import (compute_levenshtein_distance, fill_str_with_space,
                         is_correct)
from .pipelines import Compose


@DATASETS.register_module()
class KOCRDataset(Dataset):
    def __init__(
        self,
        data_root,
        img_prefix,
        ann_file,
        pipeline,
        converter,
        max_num_data=None,
        test_mode=False,
        filter_imgs=False,
        ignore_white_space=True,
        logger=None,
    ):
        """KOCR Dataset
        The dataset loads all images in 'data_root', and corresponding labels in .csv formatted 'ann_file'.
        Each line for the 'ann_file' should be comma-separated such as,{image file name},{corresponsind label}.

        Args:
            data_root (str): Path for the dataset's root directory.
            img_prefix (str): Use if the data is divided into different directory, such as 'train', 'test'.
            ann_file (str): Path for the label file which contains all the annotation info.
            pipeline (list[dict]): mmdetection-like transform object. Defaults to None.
            converter (dict): config dictionary for converter.
            test_mode (bool, optional): If True, annotation will not be loaded.
            filter_imgs (bool, optional): If True, verify if all files in the ann_file actually exist in data_root.
            ignore_white_space (bool, optional): If set as True, all white spaces in the labels will be removed.
        """
        self.img_formats = ("jpg", "jpeg", "png", "tif", "tiff", "bmp", "gif")
        self.data_root = data_root
        self.img_prefix = os.path.join(data_root, img_prefix)
        self.ann_file = os.path.join(data_root, ann_file)
        self.converter = build_converter(converter)
        self.test_mode = test_mode
        self.ignore_white_space = ignore_white_space
        self.logger = logger

        # load annotations
        self.ann_file_pattern = re.compile(r"([a-zA-Z0-9\-\_]+\.\w{3,4}),\s{0,1}(.+)")
        self.data_infos = self.load_annotations(self.ann_file)

        if max_num_data:
            random.shuffle(self.data_infos)
            self.data_infos = self.data_infos[:max_num_data]

        # filter imgs
        if filter_imgs:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file.

        Args:
            ann_file (str): A path for label file containing comma-seperated filename-label pairs.

        Returns:
            label_map (dict): A key is a filename, and a value is a label.
        """
        data_infos = []
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                match = self.ann_file_pattern.search(line)
                if match:
                    filename, label = match.group(1), match.group(2)

                    if self.ignore_white_space:
                        label = label.replace(" ", "")

                    info = dict(filename=filename, ann=label)
                    data_infos.append(info)

                elif self.logger is not None:
                    self.logger.info("Cannot find label pair for {}".format(line))

        if self.logger is not None:
            self.logger.info(
                "Loaded {} images from {}.".format(len(data_infos), self.ann_file)
            )

        return data_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        ann_info = dict(label=None)
        label_str = self.data_infos[idx]["ann"]
        label = self.converter.encode(label_str)
        ann_info["label"] = label.get("label")
        return ann_info

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results["img_prefix"] = self.img_prefix

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _filter_imgs(self):
        """Filter images not in the data_root."""
        valid_inds = []
        all_files = {}
        for dirpath, dirnames, filenames in os.walk(self.data_root):
            for filename in filenames:
                if filename.lower().endswith(self.img_formats):
                    all_files[filename] = 1

        for img_id, info in enumerate(self.data_infos):
            filename = info["filename"]
            if all_files.get(filename) is not None:
                valid_inds.append(img_id)
        return valid_inds

    def _set_group_flag(self):
        """Set flag randomly (0 or 1) in the init stage."""
        self.flag = np.random.randint(0, high=2, size=len(self), dtype=np.uint8)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                True).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

        return data

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def evaluate(
        self,
        results,
        output_type="crossentropy",
        num_show=10,
        exclude=None,
        logger=None,
        print_all=False,
    ):
        """Evaluate the results by computing accuracy.

        Args:
            results (list[dict] or dict):
                        Predictions list.
                        Each result has key 'result', and 'gt_label' (optional).
            output_type (str):
                        Evaluation method is different,
                        depending on the loss function type for the decoder.
                        ("crossentropy" or "ctc"). Defaults to "crossentropy"
            num_show (int): Number of result samples to display.
            exclude (str): Strings to be excluded in `is_correct` function. Default to None.
            logger (optional): logger. Defaults to None.
        """
        eval_method = getattr(self, "evaluate_" + output_type)

        if logger is not None:
            logger.info("Start Evaluation.")
        eval_res = eval_method(
            results,
            exclude=exclude,
            num_show=num_show,
            logger=logger,
            print_all=print_all,
        )
        return eval_res

    def evaluate_crossentropy(
        self, results, exclude=None, num_show=10, logger=None, print_all=False
    ):
        """Evaluate crossentropy version."""
        eval_res = dict(accuracy=0.0)
        num_correct = 0
        mean_dist = 0.0
        num_data = 0
        logger_cnt = 0

        for result in results:
            preds, gt_labels = result["result"], result["gt_label"]
            preds_decoded = self.converter.decode(preds)
            gt_labels = self.converter.decode(gt_labels[:, 1:])  # no "[GO]"
            filenames = result.get("filenames")

            for i, (pred, gt_label) in enumerate(zip(preds_decoded, gt_labels)):
                # compare the prediction with the gt
                if is_correct(pred, gt_label, exclude=exclude):
                    num_correct += 1
                # compute levenshtein distance
                mean_dist += compute_levenshtein_distance(
                    pred, gt_label, exclude=exclude
                )
                num_data += 1

                if logger:
                    logger_cnt = self.log(
                        pred, gt_label, exclude, logger, logger_cnt, num_show
                    )
                elif print_all:
                    if filenames:
                        filename = filenames[i]
                    else:
                        filename = None
                    self.print_console(pred, gt_label, exclude, filename)

        eval_res["accuracy"] = num_correct / num_data
        eval_res["mean_error_distance"] = mean_dist / num_data
        return eval_res

    def evaluate_ctc(
        self, results, exclude=None, num_show=10, logger=None, print_all=False
    ):
        """Evaluate ctc version."""
        eval_res = dict(accuracy=0.0)
        num_correct = 0
        mean_dist = 0.0
        num_data = 0
        logger_cnt = 0

        for result in results:
            preds, gt_labels = result["result"], result["gt_label"]
            gt_labels = self.converter.decode(gt_labels, gt=True)
            filenames = result.get("filenames")

            # decode the preds
            preds_decoded = self.converter.decode(preds)

            for i, (pred, gt_label) in enumerate(zip(preds_decoded, gt_labels)):
                # remove unk_char from gt_label
                gt_label = gt_label.replace(self.converter.unk_char, "")

                # compare the prediction with the gt
                if is_correct(pred, gt_label, exclude=exclude):
                    num_correct += 1
                # compute levenshtein distance
                mean_dist += compute_levenshtein_distance(
                    pred, gt_label, exclude=exclude
                )
                num_data += 1

                if logger:
                    logger_cnt = self.log(
                        pred, gt_label, exclude, logger, logger_cnt, num_show
                    )
                elif print_all:
                    if filenames:
                        filename = filenames[i]
                    else:
                        filename = None
                    self.print_console(pred, gt_label, exclude, filename)

        eval_res["accuracy"] = num_correct / num_data
        eval_res["mean_error_distance"] = mean_dist / num_data
        return eval_res

    def log(self, pred, gt_label, exclude, logger, logger_cnt, num_show):
        """log the comparison result."""
        if logger_cnt >= num_show:
            return logger_cnt

        if is_correct(pred, gt_label, exclude=exclude):
            correctness = "Correct!"
        else:
            correctness = "Wrong!"

        logger.info(
            "prediction: %s| ground truth: %s| %s"
            % (fill_str_with_space(pred), fill_str_with_space(gt_label), correctness,)
        )
        logger_cnt += 1

        return logger_cnt

    def print_console(self, pred, gt_label, exclude, filename):
        """log the comparison result in console."""
        if is_correct(pred, gt_label, exclude=exclude):
            correctness = "Correct!"
        else:
            correctness = "Wrong!"

        if correctness == "Wrong!":
            if filename:
                print(
                    "%s | prediction: %s| ground truth: %s| %s"
                    % (
                        filename,
                        fill_str_with_space(pred),
                        fill_str_with_space(gt_label),
                        correctness,
                    )
                )
            else:
                print(
                    "prediction: %s| ground truth: %s| %s"
                    % (
                        fill_str_with_space(pred),
                        fill_str_with_space(gt_label),
                        correctness,
                    )
                )
