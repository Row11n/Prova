from pathlib import Path
import os
import random
import copy
from .lvis_v1_categories import LVIS_CATEGORIES as LVIS_V1_CATEGORIES
from .coco import ConvertCocoPolysToMask
from . import transforms as T
import torch
from PIL import Image

def load_lvis_json(json_file, image_root, dataset_name=None):
    """
    Load a json file in LVIS's annotation format.

    Args:
        json_file (str): full path to the LVIS json annotation file.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., "lvis_v0.5_train").
            If provided, this function will put "thing_classes" into the metadata
            associated with this dataset.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from lvis import LVIS

    # json_file = PathManager.get_local_path(json_file)

    # timer = Timer()
    lvis_api = LVIS(json_file)

    meta = get_lvis_instances_meta(dataset_name)
    # sort indices for reproducible results
    img_ids = sorted(lvis_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = lvis_api.load_imgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    # Sanity check that each annotation has a unique id
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique".format(
        json_file
    )

    imgs_anns = list(zip(imgs, anns))

    def get_file_name(img_root, img_dict):
        # Determine the path including the split folder ("train2017", "val2017", "test2017") from
        # the coco_url field. Example:
        #   'coco_url': 'http://images.cocodataset.org/train2017/000000155379.jpg'
        if 'file_name' in img_dict :
            file_name = img_dict['file_name']
            split_folder = ''
        else :
            split_folder, file_name = img_dict["coco_url"].split("/")[-2:]
        return os.path.join(os.path.join(img_root, split_folder), file_name)

    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = get_file_name(image_root, img_dict)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get("not_exhaustive_category_ids", [])
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        if 'pos_category_ids' in img_dict :
            record['pos_category_ids'] = img_dict.get("pos_category_ids", []) # for custom imagenet-lvis
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.
            assert anno["image_id"] == image_id
            # obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
            obj = {"bbox": anno["bbox"]} # XYHW mode
            # LVIS data loader can be used to load COCO dataset categories. In this case `meta`
            # variable will have a field with COCO-specific category mapping.

            # FIXME: stop converting to 0-nd
            obj["category_id"] = anno["category_id"]
            # if dataset_name is not None and "thing_dataset_id_to_contiguous_id" in meta:
            #     obj["category_id"] = meta["thing_dataset_id_to_contiguous_id"][anno["category_id"]]
            # else:
            #     obj["category_id"] = anno["category_id"] - 1  # Convert 1-indexed to 0-indexed
            
            if 'segmentation' in anno :
                segm = anno["segmentation"]  # list[list[float]]
                # filter out invalid polygons (< 3 points)
                valid_segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                assert len(segm) == len(
                    valid_segm
                ), "Annotation contains an invalid polygon with < 3 points"
                assert len(segm) > 0
                obj["segmentation"] = segm
            obj['area'] = anno['area']
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts, lvis_api

def get_lvis_instances_meta(dataset_name):
    if 'v1' in dataset_name :
        return _get_lvis_v1_meta()
    if 'v3det' in dataset_name :
        return _get_v3det_meta()
    raise ValueError("No built-in metadata for dataset {}".format(dataset_name))

def _get_lvis_v1_meta():
    assert len(LVIS_V1_CATEGORIES) == 1203
    cat_ids = [k["id"] for k in LVIS_V1_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(LVIS_V1_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["synonyms"][0] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta


def _get_v3det_meta():
    from .v3det_categories import V3DET_CATEGORIES
    assert len(V3DET_CATEGORIES) == 13204
    cat_ids = [k["id"] for k in V3DET_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(V3DET_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["name"] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta

class LvisDetection():
    def __init__(self, json_file, image_root, dataset_name, transforms, return_masks, is_extra=False):
        self.data, self.lvis = load_lvis_json(json_file, image_root, dataset_name)
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self._transforms = transforms
        self.is_extra = is_extra
    
    def __len__(self):
        return len(self.data)

    @property
    def cats(self,):
        return self.lvis.cats

    def __getitem__(self, idx):
        try:
            data = self.data[idx]
        except:
            idx = random.randint(0 , len(self))
            data = self.data[idx]

        data = copy.deepcopy(data)
        anno = data.pop('annotations', [])
        img = Image.open(data['file_name']).convert('RGB')
        image_id = data['image_id']
        target = {'image_id': image_id, 'annotations': anno}
    
        img, target = self.prepare(img, target)
        
        if self._transforms is not None:
            outputs = self._transforms(img, target)
        if self.is_extra:
            img, target = outputs
            target['is_extra'] = self.is_extra
            return img, target

        return outputs


class ImagenetDetection():
    def __init__(self, json_file, image_root, dataset_name, transforms, return_masks, as_unlabeled=False):
        self.data, self.lvis = load_lvis_json(json_file, image_root, dataset_name)
        for x in self.data :
            if as_unlabeled :
                x['annotations'] = []
                continue
            h, w = x['height'], x['width']
            cat_id = x['pos_category_ids'][0]
            anno = {'bbox':[0.,0.,w,h], 'category_id':cat_id, 'area':h*w*1.}
            x['annotations'] = [anno]
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self._transforms = transforms
        self.as_unlabeled = as_unlabeled

    def __len__(self):
        return len(self.data)

    @property
    def cats(self,):
        return self.lvis.cats

    def __getitem__(self, idx, fetch_mix_result=False):
        try:
            data = self.data[idx]
        except:
            idx = random.randint(0 , len(self))
            data = self.data[idx]
        data = copy.deepcopy(data)
        anno = data.pop('annotations', [])
        img = Image.open(data['file_name']).convert('RGB')
        image_id = data['image_id']
        target = {'image_id': image_id, 'annotations': anno}

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            if isinstance(self._transforms, list) :
                for trans in self._transforms :
                    if isinstance(trans, T.Mosaic):
                        if fetch_mix_result :
                            return img, target
                        target['mix_results'] = [self.__getitem__(idx_mix, True) for idx_mix in trans.get_indexes(self.data)]
                    img, target = trans(img, target)
            else :
                img, target = self._transforms(img, target)

        target['is_extra'] = True

        return img, target


from datasets import load_dataset
import mmengine
from io import BytesIO
import base64
    
class ImagenetV3Det():
    def __init__(self, imagenet_v3det_path, cat_tree_path, transforms, return_masks, as_unlabeled=False):
        self.dataset = load_dataset(imagenet_v3det_path)['train']
        self.imgs = self.dataset['image']
        self.labels = self.dataset['label']
        self.cat_tree = mmengine.load(cat_tree_path)
        self.catid2inid = self.cat_tree['categoryid2treeid']
        self.inid2catid = {v: k for k, v in self.catid2inid.items()}
        print('init done.')
        
        # for x in self.data :
        #     if as_unlabeled :
        #         x['annotations'] = []
        #         continue
        #     h, w = x['height'], x['width']
        #     cat_id = x['pos_category_ids'][0]
        #     anno = {'bbox':[0.,0.,w,h], 'category_id':cat_id, 'area':h*w*1.}
        #     x['annotations'] = [anno]
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self._transforms = transforms
        self.as_unlabeled = as_unlabeled

    def __len__(self):
        return len(self.dataset['label'])

    # @property
    # def cats(self,):
    #     return self.lvis.cats

    def __getitem__(self, idx, fetch_mix_result=False):
        try:
            image = Image.open(BytesIO(base64.b64decode(self.imgs[idx]))).convert('RGB')
        except:
            idx = random.randint(0, len(self))
            image = Image.open(BytesIO(base64.b64decode(self.imgs[idx]))).convert('RGB')
        # data = copy.deepcopy(data)
        w, h = image.size
        cat_id = int(self.inid2catid[self.labels[idx]])
        image_id = idx
        anno = [{'bbox':[0.,0.,w,h], 'category_id':cat_id, 'area':h*w*1.}]
        target = {'image_id': image_id, 'annotations': anno}

        image, target = self.prepare(image, target)
        if self._transforms is not None:
            if isinstance(self._transforms, list) :
                for trans in self._transforms :
                    if isinstance(trans, T.Mosaic):
                        if fetch_mix_result :
                            return image, target
                        target['mix_results'] = [self.__getitem__(idx_mix, True) for idx_mix in trans.get_indexes(self.dataset['label'])]
                    image, target = trans(image, target)
            else :
                image, target = self._transforms(image, target)

        target['is_extra'] = True

        return image, target

def build(image_set, args):
    root = args.data_path
    # assert root.exists(), f'provided COCO path {root} does not exist'
    # this arg is for using partial annotation in lvis
    if args.lvis_drop_ratio > 0 :
        train_path = 'lvis_v1_train_drop0{}.json'.format(int(args.lvis_drop_ratio*10))
    else :
        train_path = 'lvis_v1_train.json'

    data_name = 'lvis_v1'
    if args.dataset_file == 'lvis':
        PATHS = {
            "train": (root , os.path.join(root, train_path)),
            "val": (root , os.path.join(root, 'lvis_v1_val.json')),
            "minival": (root, os.path.join(root, 'lvis_v1_minival.json')),
        }
    elif args.dataset_file == 'lvis_openvocab':
        PATHS = {
            "train": (root , os.path.join(root, 'lvis_v1_train_rm_rare.json')),
            "val": (root , os.path.join(root, 'lvis_v1_val.json')),
            "minival": (root, os.path.join(root, 'lvis_v1_minival.json')),
        }
    elif args.dataset_file == 'inet_lvis':
        return build_imagenet(image_set, args)
    elif args.dataset_file == 'v3det':
        PATHS = {
            "train": (root , os.path.join(root, 'v3det_2023_v1_train_filter.json')),
            "val": (root , os.path.join(root, 'v3det_2023_v1_val_tiny.json')),
            # "val": ('DATASET/v3det/tmp/V3Det' , 'DATASET/v3det/v3det_2023_v1_test_image_info.json'),
        }
        data_name = 'v3det'
    elif args.dataset_file == 'v3det_full':
        PATHS = {
            "train": (root , os.path.join(root, 'v3det_2023_v1_train_filter.json')),
            "val": (root , os.path.join(root, 'v3det_2023_v1_val.json')),
        }
        data_name = 'v3det'
    elif args.dataset_file == 'v3det_openvocab':
        PATHS = {
            "train": (root , os.path.join(root, 'v3det_2023_v1_train_ovd_base_filter.json')),
            "val": (root , os.path.join(root, 'v3det_2023_v1_val_tiny.json')),
        }
        data_name = 'v3det'
    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    try :
        inst_masks = args.inst_masks
    except :
        inst_masks = False

    from .coco import make_coco_transforms
    dataset = LvisDetection(PATHS[image_set][1], root, data_name,
                            transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args),
                            return_masks=args.masks or inst_masks or args.with_vector,
                            )

    return dataset

def build_extra_data(root, image_folder, json_file, return_masks, args):
    from .coco import make_coco_transforms
    dataset = LvisDetection(os.path.join(root, json_file), os.path.join(root, image_folder), 'lvis_v1',
                            transforms=make_coco_transforms('train', fix_size=args.fix_size, strong_aug=False, args=args),
                            return_masks=return_masks,
                            is_extra=True
                            )
    return dataset

def build_imagenet(image_set, args):
    root = args.imagenet_path
    PATHS = {
        'imagenet21k-lvis': (root, os.path.join(root, 'imagenet_lvis_image_info.json')),
    }

    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False

    from .coco import make_coco_transforms
    dataset = ImagenetDetection(PATHS['imagenet21k-lvis'][1], os.path.join(root, 'images'), 'lvis_v1',
                            transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, 
                                                            args=args, imagenet_aug=True, use_mosaic=args.imagenet_use_mosaic),
                            return_masks=args.masks,
                            as_unlabeled=args.as_unlabeled
                            )

    return dataset

def build_imagenet_v3det(image_set, args):
    imagenet_v3det_path = args.imagenet_v3det_path
    cat_tree_path = args.cat_tree_path
    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False

    from .coco import make_coco_transforms
    dataset = ImagenetV3Det(imagenet_v3det_path, cat_tree_path,
                            transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, 
                                                            args=args, imagenet_aug=True, use_mosaic=args.imagenet_use_mosaic),
                            return_masks=args.masks,
                            as_unlabeled=args.as_unlabeled
                            )

    return dataset
