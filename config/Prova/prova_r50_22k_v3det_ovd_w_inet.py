_base_ = ['baseline_4scale.py']

epochs = 24
lr_drop = 20
batch_size = 2
lr = 0.0002
enc_cls_agn=True

use_language=True
text_embed_type='visual_prototypes_v3det_ovd'
clip_align_ensemble=True
text_embed_type_2='textual_prototypes_v3det'
ensemble_with_no_clip=False

use_visual_distill=False
clip_model='RN50'
resnet_pretrain_path='path/to/your/resnet50_miil_21k.pth'

use_fed_loss=False

num_classes=13205
dn_labelbook_size=13205


use_rfs = False

use_imagenet_v3det = True
imagenet_use_mosaic = True
imagenet_v3det_path = 'path/to/your/V3Det_ImageNet21k_Cls_100'
cat_tree_path = 'path/to/your/v3det_2023_v1_category_tree.json'