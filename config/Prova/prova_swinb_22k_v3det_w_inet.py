_base_ = ['baseline_4scale.py']

backbone_dir='path/to/your/swin_B'
backbone = 'swin_B_384_22k'
use_checkpoint = True

epochs = 7
lr_drop = 6
batch_size = 2

lr = 0.0002
use_rfs=True

use_language=True
text_embed_type='visual_prototypes_v3det'
clip_align_ensemble=True
text_embed_type_2='textual_prototypes_v3det'
ensemble_with_no_clip=True
use_visual_distill=False

fed_num_sample_cats=100
num_classes=13205
dn_labelbook_size=13205

use_imagenet_v3det = True
imagenet_use_mosaic = True
imagenet_v3det_path = 'path/to/your/V3Det_ImageNet21k_Cls_100'
cat_tree_path = 'path/to/your/v3det_2023_v1_category_tree.json'
