_base_ = ['data_transformer.py']

num_classes=1204
dn_labelbook_size=1204

lr = 0.0001
param_dict_type = 'default'
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
ddetr_lr_param = False
batch_size = 2
weight_decay = 0.0001
epochs = 12
lr_drop = 11
save_checkpoint_interval = 1
eval_interval = 1
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = False
lr_drop_list = [33, 45]


modelname = 'richsem'
frozen_weights = None
backbone = 'resnet50'
use_checkpoint = False

dilation = False
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
unic_layers = 0
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = 'roi_align'
dec_layer_number = None
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_type = 'standard'
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
num_select = 300
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True
set_cost_class = 2.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 1.0
mask_loss_coef = 1.0
dice_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25

decoder_sa_type = 'sa' # ['sa', 'ca_label', 'ca_content']
matcher_type = 'HungarianMatcher' # or SimpleMinsumMatcher
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1

dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True

enc_cls_agn = False

# for dn
use_dn = True
dn_number = 100
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
embed_init_tgt = True

match_unstable_error = True

dn_labelbook_reuse_cls = True
# for ema
use_ema = False
ema_decay = 0.9997
ema_epoch = 0

use_detached_boxes_dec_out = False

use_rfs = True
rfs_repeat_sh = 0.001

use_cas = False

# for mlc
mlc_sample_number = 0
mlc_use_focal = True
mlc_loss_ceof = 1.

dn_cls_agnostic=False
mlc_type = 'normal'
use_mlc_bce=True

# lvis
lvis_drop_ratio = 0.

# imagenet
use_imagenet = False
imagenet_use_mosaic = True
imagenet_path='DATASET/imagenet-lvis'
main_weight=1
sub_weight=1
mask_bbox=False
mask_giou=False
mask_labels=False
as_unlabeled = False #treat imagenet as unlabeled data

attn_label_enc=False
inst_masks = False
inst_focal_mask_loss = False
mask_use_relative_hw = False
eval_inst_masks=False
use_query_mask = True
reuse_mask_head = True
mask_cls_agn = True
inst_masks_bce_loss=False
use_query_conv=True
mask_stride_list = [8]
is_whole_mask=False

sup_cl_loss_coef=1.
use_sup_cl = False
cl_queue_len = 64
check_pos_dn = False

add_gt=False
distill_loss_coef=0.5
distill_type = 'clip_logits'
use_language=False
use_visual_distill=False
use_mlp_proj=False
share_vl_proj=False
distill_random_boxes=False
use_clip_visual_query=False
clip_visual_resolution=224
distill_aux_layers=False
use_cnn_clip=True
clip_model='RN50'
use_cls_mlp_proj=True
two_stage_cls=False
clip_distill_objective='gt' #(gt, pred, pred_all)
use_dynamic_distill_weight = False
# federated loss
use_fed_loss=True
use_fed_on_kd = False
fed_num_sample_cats=50

# seesaw loss
use_seesaw_loss = False

# pusedo labels
use_imagenet_pusedo_labels=False

# extra_data_type
use_extra_data = False
extra_data_type = ''
clip_pusedo_th = 0.05
is_cc=False

condinst_channel_div=32
condinst_mask_channel=8
add_gt_to_mask=False
add_dn_to_mask=False
concat_mask_input=False

# vector mask
with_vector=False
n_keep=256
gt_mask_len=128
vector_loss_coef=0.7
vector_hidden_dim=256
no_vector_loss_norm=False
activation='relu'
vector_start_stage=0
loss_type='l1'
vector_mask_coef=1.

resnet_pretrain_path=''

use_tta=False