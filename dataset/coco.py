# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
if __name__=="__main__":
    # for debug only
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))

import json
from pathlib import Path
import random
import os
from PIL import Image

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

from dataset.data_util import preparing_dataset
import dataset.transforms as T
from util.box_ops import box_cxcywh_to_xyxy, box_iou

__all__ = ['build']

######################################################
# some hookers for training

class label2compat():
    def __init__(self) -> None:
        self.category_map_str = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}
        self.category_map = {int(k):v for k,v in self.category_map_str.items()}

    def __call__(self, target, img=None):
        labels = target['labels']
        res = torch.zeros(labels.shape, dtype=labels.dtype)
        for idx, item in enumerate(labels):
            res[idx] = self.category_map[item.item()] - 1
        target['label_compat'] = res
        if img is not None:
            return target, img
        else:
            return target

class label_compat2onehot():
    def __init__(self, num_class=80, num_output_objs=1):
        self.num_class = num_class
        self.num_output_objs = num_output_objs
        if num_output_objs != 1:
            raise DeprecationWarning("num_output_objs!=1, which is only used for comparison")

    def __call__(self, target, img=None):
        labels = target['label_compat']
        place_dict = {k:0 for k in range(self.num_class)}
        if self.num_output_objs == 1:
            res = torch.zeros(self.num_class)
            for i in labels:
                itm = i.item()
                res[itm] = 1.0
        else:
            # compat with baseline
            res = torch.zeros(self.num_class, self.num_output_objs)
            for i in labels:
                itm = i.item()
                res[itm][place_dict[itm]] = 1.0
                place_dict[itm] += 1
        target['label_compat_onehot'] = res
        if img is not None:
            return target, img
        else:
            return target

class box_label_catter():
    def __init__(self):
        pass

    def __call__(self, target, img=None):
        labels = target['label_compat']
        boxes = target['boxes']
        box_label = torch.cat((boxes, labels.unsqueeze(-1)), 1)
        target['box_label'] = box_label
        if img is not None:
            return target, img
        else:
            return target

def label2onehot(label, num_classes):
    """
    label: Tensor(K)
    """
    res = torch.zeros(num_classes)
    for i in label:
        itm = int(i.item())
        res[itm] = 1.0
    return res

class RandomSelectBoxlabels():
    def __init__(self, num_classes, leave_one_out=False, blank_prob=0.8,
                    prob_first_item = 0.0,
                    prob_random_item = 0.0,
                    prob_last_item = 0.8,
                    prob_stop_sign = 0.2
                ) -> None:
        self.num_classes = num_classes
        self.leave_one_out = leave_one_out
        self.blank_prob = blank_prob

        self.set_state(prob_first_item, prob_random_item, prob_last_item, prob_stop_sign)
        

    def get_state(self):
        return [self.prob_first_item, self.prob_random_item, self.prob_last_item, self.prob_stop_sign]

    def set_state(self, prob_first_item, prob_random_item, prob_last_item, prob_stop_sign):
        sum_prob = prob_first_item + prob_random_item + prob_last_item + prob_stop_sign
        assert sum_prob - 1 < 1e-6, \
            f"Sum up all prob = {sum_prob}. prob_first_item:{prob_first_item}" \
            + f"prob_random_item:{prob_random_item}, prob_last_item:{prob_last_item}" \
            + f"prob_stop_sign:{prob_stop_sign}"

        self.prob_first_item = prob_first_item
        self.prob_random_item = prob_random_item
        self.prob_last_item = prob_last_item
        self.prob_stop_sign = prob_stop_sign
        

    def sample_for_pred_first_item(self, box_label: torch.FloatTensor):
        box_label_known = torch.Tensor(0,5)
        box_label_unknown = box_label
        return box_label_known, box_label_unknown

    def sample_for_pred_random_item(self, box_label: torch.FloatTensor):
        n_select = int(random.random() * box_label.shape[0])
        box_label = box_label[torch.randperm(box_label.shape[0])]
        box_label_known = box_label[:n_select]
        box_label_unknown = box_label[n_select:]
        return box_label_known, box_label_unknown

    def sample_for_pred_last_item(self, box_label: torch.FloatTensor):
        box_label_perm = box_label[torch.randperm(box_label.shape[0])]
        known_label_list = []
        box_label_known = []
        box_label_unknown = []
        for item in box_label_perm:
            label_i = item[4].item()
            if label_i in known_label_list:
                box_label_known.append(item)
            else:
                # first item
                box_label_unknown.append(item)
                known_label_list.append(label_i)
        box_label_known = torch.stack(box_label_known) if len(box_label_known) > 0 else torch.Tensor(0,5)
        box_label_unknown = torch.stack(box_label_unknown) if len(box_label_unknown) > 0 else torch.Tensor(0,5)
        return box_label_known, box_label_unknown

    def sample_for_pred_stop_sign(self, box_label: torch.FloatTensor):
        box_label_unknown = torch.Tensor(0,5)
        box_label_known = box_label
        return box_label_known, box_label_unknown

    def __call__(self, target, img=None):
        box_label = target['box_label'] # K, 5

        dice_number = random.random()

        if dice_number < self.prob_first_item:
            box_label_known, box_label_unknown = self.sample_for_pred_first_item(box_label)
        elif dice_number < self.prob_first_item + self.prob_random_item:
            box_label_known, box_label_unknown = self.sample_for_pred_random_item(box_label)
        elif dice_number < self.prob_first_item + self.prob_random_item + self.prob_last_item:
            box_label_known, box_label_unknown = self.sample_for_pred_last_item(box_label)
        else:
            box_label_known, box_label_unknown = self.sample_for_pred_stop_sign(box_label)

        target['label_onehot_known'] = label2onehot(box_label_known[:,-1], self.num_classes)
        target['label_onehot_unknown'] = label2onehot(box_label_unknown[:, -1], self.num_classes)
        target['box_label_known'] = box_label_known
        target['box_label_unknown'] = box_label_unknown

        return target, img


class RandomDrop():
    def __init__(self, p=0.2) -> None:
        self.p = p

    def __call__(self, target, img=None):
        known_box = target['box_label_known']
        num_known_box = known_box.size(0)
        idxs = torch.rand(num_known_box)
        # indices = torch.randperm(num_known_box)[:int((1-self).p*num_known_box + 0.5 + random.random())]
        target['box_label_known'] = known_box[idxs > self.p]
        return target, img


class BboxPertuber():
    def __init__(self, max_ratio = 0.02, generate_samples = 1000) -> None:
        self.max_ratio = max_ratio
        self.generate_samples = generate_samples
        self.samples = self.generate_pertube_samples()
        self.idx = 0

    def generate_pertube_samples(self):
        import torch
        samples = (torch.rand(self.generate_samples, 5) - 0.5) * 2 * self.max_ratio
        return samples

    def __call__(self, target, img):
        known_box = target['box_label_known'] # Tensor(K,5), K known bbox
        K = known_box.shape[0]
        known_box_pertube = torch.zeros(K, 6) # 4:bbox, 1:prob, 1:label
        if K == 0:
            pass
        else:
            if self.idx + K > self.generate_samples:
                self.idx = 0
            delta = self.samples[self.idx: self.idx + K, :]
            known_box_pertube[:, :4] = known_box[:, :4] + delta[:, :4]
            iou = (torch.diag(box_iou(box_cxcywh_to_xyxy(known_box[:, :4]), box_cxcywh_to_xyxy(known_box_pertube[:, :4]))[0])) * (1 + delta[:, -1])
            known_box_pertube[:, 4].copy_(iou)
            known_box_pertube[:, -1].copy_(known_box[:, -1])

        target['box_label_known_pertube'] = known_box_pertube
        return target, img


class RandomCutout():
    def __init__(self, factor=0.5) -> None:
        self.factor = factor

    def __call__(self, target, img=None):
        unknown_box = target['box_label_unknown']           # Ku, 5
        known_box = target['box_label_known_pertube']       # Kk, 6
        Ku = unknown_box.size(0)

        known_box_add = torch.zeros(Ku, 6) # Ku, 6
        known_box_add[:, :5] = unknown_box
        known_box_add[:, 5].uniform_(0.5, 1) 
        

        known_box_add[:, :2] += known_box_add[:, 2:4] * (torch.rand(Ku, 2) - 0.5) / 2
        known_box_add[:, 2:4] /= 2

        target['box_label_known_pertube'] = torch.cat((known_box, known_box_add))
        return target, img




class RandomSelectBoxes():
    def __init__(self, num_class=80) -> None:
        Warning("This is such a slow function and will be deprecated soon!!!")
        self.num_class = num_class

    def __call__(self, target, img=None):
        boxes = target['boxes']
        labels = target['label_compat']

        # transform to list of tensors
        boxs_list = [[] for i in range(self.num_class)]
        for idx, item in enumerate(boxes):
            label = labels[idx].item()
            boxs_list[label].append(item)
        boxs_list_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in boxs_list]

        # random selection
        box_known = []
        box_unknown = []
        for idx, item in enumerate(boxs_list_tensor):
            ncnt = item.shape[0]
            nselect = int(random.random() * ncnt) # close in both sides, much faster than random.randint
            # import ipdb; ipdb.set_trace()
            item = item[torch.randperm(ncnt)]
            # random.shuffle(item)
            box_known.append(item[:nselect])
            box_unknown.append(item[nselect:])
        # import ipdb; ipdb.set_trace()
        # box_known_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in box_known]
        # box_unknown_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in box_unknown]
        # print('box_unknown_tensor:', box_unknown_tensor)
        target['known_box'] = box_known
        target['unknown_box'] = box_unknown
        return target, img






        


# class BoxCatter():
#     def __init__(self) -> None:
#         pass

#     def __call__(self, target, img):
#         """
#         known_box_cat:
#             - Tensor(k*5), 
#                 * Tensor[:, :4]: bbox,  
#                 * Tensor[:, -1]: label
#         """
#         known_box = target['known_box']
#         boxes_list = []
#         for idx, boxes in enumerate(known_box):
#             nbox = boxes.shape[0]
#             boxes_idx = torch.cat([boxes, torch.Tensor([idx] * nbox).unsqueeze(1)], 1)
#             boxes_list.append(boxes_idx)
#         known_box_cat = torch.cat(boxes_list, 0)
#         target['known_box_cat'] = known_box_cat
#         return target, img
        

class MaskCrop():
    def __init__(self) -> None:
        pass

    def __call__(self, target, img):
        known_box = target['known_box']
        h,w = img.shape[1:] # h,w
        # imgsize = target['orig_size'] # h,w
        # import ipdb; ipdb.set_trace()
        scale = torch.Tensor([w, h, w, h])

        # _cnt = 0
        for boxes in known_box:
            if boxes.shape[0] == 0:
                continue
            box_xyxy = box_cxcywh_to_xyxy(boxes) * scale
            for box in box_xyxy:
                x1, y1, x2, y2 = [int(i) for i in box.tolist()]
                img[:, y1:y2, x1:x2] = 0
                # _cnt += 1
        # print("_cnt:", _cnt)
        return target, img



        

dataset_hook_register = {
    'label2compat': label2compat,
    'label_compat2onehot': label_compat2onehot,
    'box_label_catter': box_label_catter,
    'RandomSelectBoxlabels': RandomSelectBoxlabels,
    'RandomSelectBoxes': RandomSelectBoxes,
    'MaskCrop': MaskCrop,
    'BboxPertuber': BboxPertuber,
}
                
##################################################################################

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, aux_target_hacks=None, is_extra=False):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.aux_target_hacks = aux_target_hacks
        self.is_extra = is_extra

    def change_hack_attr(self, hackclassname, attrkv_dict):
        target_class = dataset_hook_register[hackclassname]
        for item in self.aux_target_hacks:
            if isinstance(item, target_class):
                for k,v in attrkv_dict.items():
                    setattr(item, k, v)

    def get_hack(self, hackclassname):
        target_class = dataset_hook_register[hackclassname]
        for item in self.aux_target_hacks:
            if isinstance(item, target_class):
                return item

    @property
    def cats(self,):
        return self.coco.cats

    def _super_get_item(self, index: int) :
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        if isinstance(img_id, str):
            img_id = [img_id]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


    def __getitem__(self, idx, fetch_mix_result=False):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        try:
            img, target = self._super_get_item(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = self._super_get_item(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        if self._transforms is not None:
            if isinstance(self._transforms, list) :
                for trans in self._transforms :
                    if isinstance(trans, T.Mosaic):
                        if fetch_mix_result :
                            return img, target
                        target['mix_results'] = [self.__getitem__(idx_mix, True) for idx_mix in trans.get_indexes(self)]
                    img, target = trans(img, target)
            else :
                img, target = self._transforms(img, target)

        # convert to needed format
        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                target, img = hack_runner(target, img=img)
        if self.is_extra:
            target['is_extra'] = self.is_extra

        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        if not isinstance(image_id, str):
            image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None, imagenet_aug=False, use_mosaic=True):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]
    
    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i*data_aug_scale_overlap) for i in scales]
        max_size = int(max_size*data_aug_scale_overlap)
        scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]
    # else:
    #     scales = getattr(args, 'data_aug_scales', scales)
    #     max_size = getattr(args, 'data_aug_max_size', max_size)
    #     scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    #     scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    datadict_for_print = {
        'scales': scales,
        'max_size': max_size,
        'scales2_resize': scales2_resize,
        'scales2_crop': scales2_crop
    }
    print("data_aug_params:", json.dumps(datadict_for_print, indent=2))
        

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                # T.RandomResize([(512, 512)]),
                normalize,
            ])
        
        # if os.environ.get('IPDB_DEBUG_SHILONG') == 'INFO':
        #     import datasets.sltransform as SLT
        #     return T.Compose([
        #         T.RandomHorizontalFlip(),
        #         T.RandomSelect(
        #             T.RandomResize(scales, max_size=1333),
        #             T.Compose([
        #                 T.RandomResize([400, 500, 600]),
        #                 T.RandomSizeCrop(384, 600),
        #                 T.RandomResize(scales, max_size=1333),
        #             ])
        #         ),
        #         SLT.RandomCropDebug(),
        #         SLT.LightingNoise(),
        #         SLT.AdjustBrightness(2),
        #         SLT.AdjustContrast(2),
        #         SLT.Albumentations(),
        #         normalize,
        #     ])

        # if strong_aug:
        #     import datasets.sltransform as SLT
        #     return T.Compose([
        #         T.RandomHorizontalFlip(),
        #         T.RandomSelect(
        #             T.RandomResize(scales, max_size=max_size),
        #             T.Compose([
        #                 T.RandomResize(scales2_resize),
        #                 T.RandomSizeCrop(*scales2_crop),
        #                 T.RandomResize(scales, max_size=max_size),
        #             ])
        #         ),
        #         T.RandomSelect(
        #             SLT.RandomSelectMulti([
        #                 SLT.RandomCrop(),
        #                 SLT.LightingNoise(),
        #                 SLT.AdjustBrightness(2),
        #                 SLT.AdjustContrast(2),
        #             ]),                   
        #             SLT.Albumentations(),
        #             p=0.05
        #         ),
        #         normalize,
        #     ])

        if strong_aug:
            import dataset.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    # SLT.Rotate(10),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),              
                # # for debug only  
                # SLT.RandomCrop(),
                # SLT.LightingNoise(),
                # SLT.AdjustBrightness(2),
                # SLT.AdjustContrast(2),
                # SLT.Rotate(10),
                normalize,
            ])
        
        if imagenet_aug :
            return [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
            ] + ([T.Mosaic()] if use_mosaic else []) + [normalize,]

        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'eval_debug', 'train_reg', 'test']:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   
        
        if args.use_tta:
            tta_scale_min = getattr(args, 'testtime_aug_scale_min', 1200)
            tta_scale_max = getattr(args, 'testtime_aug_scale_max', 1200)
            tta_scale_step = getattr(args, 'testtime_aug_scale_step', 1)
            step = (tta_scale_max - tta_scale_min) / tta_scale_step
            steps = [int(tta_scale_min + step * i) for i in range(tta_scale_step+1)]
            tta_scales = getattr(args, 'testtime_aug_scales', steps)
            tta_max_size = getattr(args, 'testtime_aug_max_size', max_size)
            return T.ComposeTTA([
                # T.HorizontalFlip(), 
                # T.RandomHorizontalFlip(),
                T.ExhaustiveResize(tta_scales, max_size=tta_max_size),
                normalize,
            ])    

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])



    raise ValueError(f'unknown {image_set}')


def get_aux_target_hacks_list(image_set, args):
    if args.modelname in ['q2bs_mask', 'q2bs']:
        aux_target_hacks_list = [
            label2compat(), 
            label_compat2onehot(), 
            RandomSelectBoxes(num_class=args.num_classes)
        ]
        if args.masked_data and image_set == 'train':
            # aux_target_hacks_list.append()
            aux_target_hacks_list.append(MaskCrop())
    elif args.modelname in ['q2bm_v2', 'q2bs_ce', 'q2op', 'q2ofocal', 'q2opclip', 'q2ocqonly']:
        aux_target_hacks_list = [
            label2compat(),
            label_compat2onehot(),
            box_label_catter(),
            RandomSelectBoxlabels(num_classes=args.num_classes,
                                    prob_first_item=args.prob_first_item,
                                    prob_random_item=args.prob_random_item,
                                    prob_last_item=args.prob_last_item,
                                    prob_stop_sign=args.prob_stop_sign,
                                    ),
            BboxPertuber(max_ratio=0.02, generate_samples=1000),
        ]
    elif args.modelname in ['q2omask', 'q2osa']:
        if args.coco_aug:
            aux_target_hacks_list = [
                label2compat(),
                label_compat2onehot(),
                box_label_catter(),
                RandomSelectBoxlabels(num_classes=args.num_classes,
                                        prob_first_item=args.prob_first_item,
                                        prob_random_item=args.prob_random_item,
                                        prob_last_item=args.prob_last_item,
                                        prob_stop_sign=args.prob_stop_sign,
                                        ),
                RandomDrop(p=0.2),
                BboxPertuber(max_ratio=0.02, generate_samples=1000),
                RandomCutout(factor=0.5)
            ]
        else:
            aux_target_hacks_list = [
                label2compat(),
                label_compat2onehot(),
                box_label_catter(),
                RandomSelectBoxlabels(num_classes=args.num_classes,
                                        prob_first_item=args.prob_first_item,
                                        prob_random_item=args.prob_random_item,
                                        prob_last_item=args.prob_last_item,
                                        prob_stop_sign=args.prob_stop_sign,
                                        ),
                BboxPertuber(max_ratio=0.02, generate_samples=1000),
            ]
    else:
        aux_target_hacks_list = None

    return aux_target_hacks_list


from torchvision.datasets import ImageFolder
class ImageFolderDetection():
    def __init__(self, image_root, json_file, transforms, return_masks, is_extra=False,
                    is_cc=False):
        self.image_root = image_root
        with open(json_file) as f:
            self.data = json.load(f)
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self._transforms = transforms
        self.is_extra = is_extra
        self.is_cc = is_cc

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, index: int, fetch_mix_result=False):
        if self.is_cc :
            anno_dict = self.data[index]
            caption, path = anno_dict['caption'], anno_dict['image']
            target = 0
        else :
            path, target = self.data[index]
        path = os.path.join(self.image_root, path)

        img = Image.open(path).convert('RGB')
        image_id = index  # placeholder
        w, h = img.size
        cat_id = target
        anno = [{'bbox':[0.,0.,w,h], 'category_id':cat_id, 'area':h*w*1.}]
        target = {'image_id': image_id, 'annotations': anno}

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            for trans in self._transforms :
                if isinstance(trans, T.Mosaic):
                    if fetch_mix_result :
                        return img, target
                    target['mix_results'] = [self.__getitem__(idx_mix, True) for idx_mix in trans.get_indexes(self.data)]
                img, target = trans(img, target)
        if self.is_extra:
            target['is_extra'] = self.is_extra
        return img, target


def build_extra_cocostyle_data(root, image_folder, json_file, return_masks, args,
                               is_image_folder=False, is_image_aug=False, image_set='train'):
    from .coco import make_coco_transforms
    img_folder = os.path.join(root, image_folder)
    json_file = os.path.join(root, json_file)
    if is_image_folder :
        dataset = ImageFolderDetection(img_folder, json_file,
                                    transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=False, args=args, 
                                                        imagenet_aug=True, use_mosaic=args.imagenet_use_mosaic),
                                    return_masks=return_masks,
                                    is_extra=True, is_cc=args.is_cc)
    else :
        dataset = CocoDetection(img_folder, json_file,
                                transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=False, args=args,
                                                        imagenet_aug=is_image_aug, use_mosaic=args.imagenet_use_mosaic),
                                return_masks=return_masks,
                                is_extra=True
                                )
    return dataset

def build(image_set, args):
    root = Path(args.data_path)
    # assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    if args.dataset_file == 'object365':
        PATHS = {
            "train": (root / 'train' , root / 'objects365_train.json'),
            "val": (root / 'val' , root /'objects365_val.json'),
        }
    elif args.dataset_file == 'v3det':
        PATHS = {
            "train": (root , root / 'v3det_2023_v1_train.json'),
            "val": (root , root /'v3det_2023_v1_val_tiny.json'),
            # "val": ('DATASET/v3det/tmp/V3Det' , 'DATASET/v3det/v3det_2023_v1_test_image_info.json'),
        }
    elif args.dataset_file == 'v3det_full':
        PATHS = {
            "train": (root , root / 'v3det_2023_v1_train.json'),
            "val": (root , root /'v3det_2023_v1_val.json'),
            # "val": ('DATASET/v3det/tmp/V3Det' , 'DATASET/v3det/v3det_2023_v1_test_image_info.json'),
        }
    elif args.dataset_file == 'v3det_test':
        PATHS = {
            "train": (root , root / 'v3det_2023_v1_train.json'),
            "val": ('DATASET/v3det/tmp/V3Det' , 'DATASET/v3det/v3det_2023_v1_test_image_info.json'),
        }
    elif args.dataset_file == 'v3det_openvocab':
        PATHS = {
            "train": (root , root / 'v3det_2023_v1_train_ovd_base.json'),
            "val": (root , root /'v3det_2023_v1_val_tiny.json'),
            # "val": ('DATASET/v3det/tmp/V3Det' , 'DATASET/v3det/v3det_2023_v1_test_image_info.json'),
        }
    elif args.dataset_file == 'vg':
        PATHS = {
            "train": (root / "images", root / "annotations/visual_genome_train.json"),
            "val": (root / 'images' , root /'annotations/visual_genome_val.json'),
        }
    elif args.dataset_file == 'oid':
        PATHS = {
            "train": (root / "train", root / "openimages_challenge_2019_train_bbox.json"),
            "val": (root / "val" , root /"openimages_challenge_2019_val_bbox.json"),
        }
    # elif args.dataset_file == 'inet':
    #     dataset = build_extra_cocostyle_data(root, 'unzip', 'images.json', False, args, is_image_folder=True, image_set=image_set)
    #     return dataset
    else :
        PATHS = {
            "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "train_reg": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
            "eval_debug": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
            "test": (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json' ),
        }

    # add some hooks to datasets
    aux_target_hacks_list = get_aux_target_hacks_list(image_set, args)
    img_folder, ann_file = PATHS[image_set]

    # copy to local path
    if os.environ.get('DATA_COPY_SHILONG') == 'INFO':
        preparing_dataset(dict(img_folder=img_folder, ann_file=ann_file), image_set, args)

    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    try :
        inst_masks = args.inst_masks
    except :
        inst_masks = False
    dataset = CocoDetection(img_folder, ann_file, 
            transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args), 
            return_masks=args.masks or inst_masks or args.with_vector,
            aux_target_hacks=aux_target_hacks_list,
        )

    return dataset
