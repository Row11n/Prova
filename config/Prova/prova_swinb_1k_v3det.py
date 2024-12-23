_base_ = ['baseline_4scale.py']

backbone_dir='path/to/your/swin_B'
backbone = 'swin_B_224_1k'
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


