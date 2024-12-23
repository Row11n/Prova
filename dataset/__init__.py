# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .lvis import LvisDetection

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    if isinstance(dataset, LvisDetection):
        return dataset.lvis


def build_dataset(image_set, args, imagenet_lvis=False, imagenet_v3det=False):
    if imagenet_lvis :
        from .lvis import build_imagenet
        return build_imagenet(image_set, args)
    if imagenet_v3det:
        from .lvis import build_imagenet_v3det
        return build_imagenet_v3det(image_set, args)
    if args.dataset_file in ('coco', 'object365', 'vg', 'oid', 'inet', 'v3det_test'):
        return build_coco(image_set, args)
    if args.dataset_file in ('lvis', 'inet_lvis', 'lvis_openvocab', 'v3det', 'v3det_full', 'v3det_openvocab'):
        from .lvis import build as build_lvis
        return build_lvis(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'o365':
        from .o365 import build_o365_combine
        return build_o365_combine(image_set, args)
    if args.dataset_file == 'vanke':
        from .vanke import build_vanke
        return build_vanke(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
