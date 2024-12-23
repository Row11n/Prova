# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
from collections import defaultdict
from functools import partial
import itertools
import os
import contextlib
import copy
import numpy as np
import torch
import logging
from tabulate import tabulate

from lvis import LVIS, LVISEval, LVISResults

from util.misc import all_gather
import pycocotools.mask as mask_util

def create_small_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table

def _evaluate_predictions_on_lvis(lvis_gt, lvis_results, iou_type, use_cats=True):
    """
    Args:
        iou_type (str):
        kpt_oks_sigmas (list[float]):
        class_names (None or list[str]): if provided, will use it to predict
            per-category AP.

    Returns:
        a dict of {metric name: score}
    """
    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
    }[iou_type]

    logger = logging.getLogger(__name__)

    if len(lvis_results) == 0:  # TODO: check if needed
        logger.warn("No predictions from the model!")
        return {metric: float("nan") for metric in metrics}

    if iou_type == "segm":
        lvis_results = copy.deepcopy(lvis_results)
        # When evaluating mask AP, if the results contain bbox, LVIS API will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in lvis_results:
            c.pop("bbox", None)

    from lvis import LVISEval, LVISResults

    lvis_results = LVISResults(lvis_gt, lvis_results)
    lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type)
    lvis_eval.params.use_cats = use_cats
    lvis_eval.run()
    lvis_eval.print_results()

    # Pull the standard metrics from the LVIS results
    results = lvis_eval.get_results()
    results = {metric: float(results[metric] * 100) for metric in metrics}
    # logger.info("Evaluation results for {}: \n".format(iou_type) + create_small_table(results))
    return lvis_eval, results


class LvisEvaluator(object):
    def __init__(self, coco_gt, iou_types, useCats=True):
        use_cats = useCats
        assert isinstance(iou_types, (list, tuple))
        # coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = partial(LVISEval, iou_type=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.use_cats = use_cats
        # self.results = defaultdict(list)
        self.predictions = defaultdict(list)
        self._result = defaultdict(list)
        self.logger = logging.getLogger('detr')

    def update(self, predictions):
        for iou_type in self.iou_types :
            pred = self.prepare(predictions, iou_type=iou_type)
            self.predictions[iou_type].extend(pred)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types :
            predictions = all_gather(self.predictions[iou_type])
            
            predictions = list(itertools.chain(*predictions))
            self.coco_eval[iou_type], self._result[iou_type] = _evaluate_predictions_on_lvis(self.coco_gt, predictions, iou_type, self.use_cats)
    
    def accumulate(self):
        pass

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            self.logger.info("IoU metric: {}".format(iou_type))
            coco_eval.stats = np.array(list(coco_eval.results.values()))
            self.logger.info("Evaluation results for {}: \n".format(iou_type) + create_small_table (coco_eval.results))

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            if not isinstance(prediction["scores"], list):
                scores = prediction["scores"].tolist()
            else:
                scores = prediction["scores"]
            if not isinstance(prediction["labels"], list):
                labels = prediction["labels"].tolist()
            else:
                labels = prediction["labels"]

        
            try:
                coco_results.extend(
                    [
                        {
                            "image_id": original_id,
                            "category_id": labels[k],
                            "bbox": box,
                            "score": scores[k],
                        }
                        for k, box in enumerate(boxes)
                    ]
                )
            except:
                import ipdb; ipdb.set_trace()
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
