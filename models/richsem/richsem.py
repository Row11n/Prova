import copy
from functools import partial
import numpy as np
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms, batched_nms

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid, all_gather)

from .backbone import build_backbone
from .matcher import build_matcher, build_matcher4tta
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
from .deformable_transformer import build_deformable_transformer
from .utils import sigmoid_focal_loss, MLP

from ..registry import MODULE_BUILD_FUNCS
from .dn_components import prepare_for_cdn,dn_post_process
from detectron2.layers.roi_align import ROIAlign
import clip
from .clip_text_encoder import build_model as build_clip_model
from clip.utils import get_prompt_templates
import dataset.transforms as T

class LabelEnc():
    def __init__(self, weight) -> None:
        self.weight = weight

    def __call__(self, m):
        return F.embedding(m, self.weight)

class CLIPAlign(nn.Module):
    def __init__(self, v_dim, freeze=True, use_cnn_clip=True, clip_model='RN50',
                 use_label_enc=False, use_visual=False,
                 use_mlp_cls=False, use_mlp_distill=False,
                 share_vl_proj=False, use_clip_visual_proj=False,
                 text_embed_type="clip", ensemble=False, ensemble_text_embed_type="",
                 ensemble_with_no_clip=False, num_classes=13205,
                 debug=False):
        super().__init__()
        if use_cnn_clip :
            self.clip = build_clip_model(clip_model, not_use_visual=False)
        else :
            self.clip = build_clip_model("ViT-B/32", not_use_visual=False)
        self.clip.train()
        self.use_cnn_clip = use_cnn_clip
        self.freeze = freeze
        if freeze :
            for p in self.clip.parameters():
                p.requires_grad = False
            self.clip.eval()

        offline_text_embed_path_dict = {
            'visual_prototypes_v3det': "./prototypes/visual_prototypes_v3det.pt",
            'textual_prototypes_v3det': "./prototypes/textual_prototypes_v3det.pt",
            'visual_prototypes_v3det_ovd': "./prototypes/visual_prototypes_v3det_ovd.pt",
        }

        # l_dim = self.clip.ln_final.weight.shape[0]
        text_weight_1 = torch.load(offline_text_embed_path_dict[text_embed_type], map_location='cpu')
        l_dim = text_weight_1.shape[-1]
        self.logit_scale = self.clip.logit_scale
        # freeze the logit_scale to avoid sub-optimal
        self.logit_scale.requires_grad = False
        # self.logit_scale.requires_grad = True

        self.share_vl_proj = share_vl_proj

        if share_vl_proj :
            vl_proj = MLP(v_dim, v_dim, l_dim, 4)
            nn.init.normal_(vl_proj.layers[-1].weight, std=l_dim ** -0.5)
            nn.init.constant_(vl_proj.layers[-1].bias.data, 0)
            self.text_proj = None
            self.dino_visual_proj = vl_proj
            self.clip_visual_proj = vl_proj
        else :
            proj_mlp = MLP(v_dim, v_dim, l_dim, 4)
            nn.init.normal_(proj_mlp.layers[-1].weight, std=l_dim ** -0.5)
            nn.init.constant_(proj_mlp.layers[-1].bias.data, 0)
            proj_linear = nn.Linear(v_dim, l_dim, bias=False)
            nn.init.normal_(proj_linear.weight, std=l_dim ** -0.5)
            self.text_proj = None
            if use_mlp_cls :
                self.dino_visual_proj = copy.deepcopy(proj_mlp)
            else :
                self.dino_visual_proj = copy.deepcopy(proj_linear)
            self.ensemble = ensemble
            self.ensemble_with_no_clip = ensemble_with_no_clip
            self.num_classes = num_classes
            if use_visual and use_clip_visual_proj:
                if use_mlp_distill :
                    self.clip_visual_proj = copy.deepcopy(proj_mlp)
                else :
                    self.clip_visual_proj = copy.deepcopy(proj_linear)
        if use_label_enc :
            self.label_proj = nn.Linear(l_dim, v_dim, bias=False)
            nn.init.normal_(self.label_proj.weight, std=l_dim ** -0.5)

        self.text_embed = None
        if debug:
            import pdb; pdb.set_trace()
        self.text_embed_type = text_embed_type
        assert self.text_embed_type in offline_text_embed_path_dict
        self.text_embed_path = offline_text_embed_path_dict[self.text_embed_type]
        
        if self.ensemble:
            self.text_embed_path_2 = offline_text_embed_path_dict[ensemble_text_embed_type]
            text_embed_2 = torch.load(self.text_embed_path_2, map_location=torch.device('cpu'))
            self.text_embed_2 = text_embed_2.cuda()
            l_dim_2 = text_embed_2.shape[-1]
            proj_linear_2 = nn.Linear(v_dim, l_dim_2, bias=False)
            nn.init.normal_(proj_linear_2.weight, std=l_dim_2 ** -0.5)
            self.dino_visual_proj_2 = copy.deepcopy(proj_linear_2)
            if self.ensemble_with_no_clip:
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                self.linear_classifier = nn.Linear(v_dim, self.num_classes)
                self.linear_classifier.bias.data = torch.ones(self.num_classes) * bias_value
        
        self.debug = debug

    @property
    def input_resolution(self):
        return self.clip.visual.input_resolution

    @property
    def patch_size(self):
        return self.clip.visual.patch_size

    @property
    def grid_size(self):
        return self.clip.visual.input_resolution // self.clip.visual.patch_size

    def set_clip_mode(self):
        if self.freeze :
            self.clip.eval()

    def clip_vl_logits(self, visual_embed, requires_grad=False):
        if requires_grad :
            text_feature = self._get_text_features(visual_embed.device, None)
            clip_logits = visual_embed @ text_feature.T
            logit_scale = self.logit_scale.exp()
            # (b, nq, c) @ (num_c, c)
            clip_logits = logit_scale * torch.matmul(visual_embed, text_feature.transpose(-1,-2))
        else :
            with torch.no_grad():
                text_feature = self._get_text_features(visual_embed.device, None)
                clip_logits = visual_embed @ text_feature.T
                logit_scale = self.logit_scale.exp()
                # (b, nq, c) @ (num_c, c)
                clip_logits = logit_scale * torch.matmul(visual_embed, text_feature.transpose(-1,-2))
        return clip_logits

    def set_total_text(self, cats):
        if self.text_embed is not None :
            return self.text_embed
        import os
        if self.debug:
            import pdb; pdb.set_trace()
        if os.path.exists(self.text_embed_path):
            print("Loading offline cat embed...")
            text_embed = torch.load(self.text_embed_path, map_location=torch.device('cpu'))
            # current_device = torch.cuda.current_device()
            # self.text_embed = text_embed.cuda(current_device)
            self.text_embed = text_embed.cuda()
            return self.text_embed
        def convert_cats_to_text(cats, prompt):
            text_list = []
            count = 0
            max_id = max(cats.keys())
            for id in range(max_id+1):
                cat = cats.get(id, {'name':'none'})['name']
                if cat != 'none':
                    cat = prompt.format(cat)
                text_list.append(cat)
            return text_list
        prompt_list = get_prompt_templates()
        total_text_list = [convert_cats_to_text(cats, prompt) for prompt in prompt_list]

        text_embed_list = []
        from tqdm import tqdm
        if self.debug:
            import pdb; pdb.set_trace()
        with torch.no_grad():
            for total_text in tqdm(total_text_list):
                ltt = len(total_text)
                text_embed = []
                for j in tqdm(range(0, ltt, 1000)):
                    tt = total_text[j:min(j+1000, ltt)]
                    tt = clip.tokenize(tt).cuda()
                    t_emb = self.clip.encode_text(tt)
                    text_embed.append(t_emb)
                text_embed = torch.cat(text_embed, dim=0)
                text_embed_list.append(text_embed)
        self.text_embed = torch.stack(text_embed_list).mean(0)
        torch.cuda.empty_cache()
        return self.text_embed

    def proj_dino_hs(self, hidden_features):
        hidden_features = self.clip_visual_proj(hidden_features)
        return hidden_features

    def clip_encode_img(self, imgs):
        with torch.no_grad():
            image_features = self.clip.encode_image(imgs)
        return image_features

    def _get_text_features(self, device, proj):
        def proj_text(text_embed):
            if proj :
                return proj(text_embed)
            return text_embed
        if self.text_embed is not None :
            text_embed = self.text_embed
            text_features = proj_text(text_embed)
        else :
            raise NotImplementedError
        return text_features

    @property
    def visual_input_size(self):
        return self.clip.visual.input_resolution

    def get_label_enc(self, m):
        text_feature = self._get_text_features(m.device, proj=self.label_proj)
        return F.embedding(m, text_feature)

    def forward(self, visual_embed):
        return self.forward_hs([visual_embed])[0]

    def forward_hs(self, hs):
        text_features = self._get_text_features(hs[0].device, proj=self.text_proj)
        logit_scale = self.logit_scale.exp()

        logits_list = []
        for visual_embed in hs :
            image_features = self.dino_visual_proj(visual_embed)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) #(bs, nq, c)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)    #(bs, c)
            # logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_image = logit_scale * torch.matmul(image_features, text_features.transpose(-1,-2))
            if self.ensemble:
                image_features_2 = self.dino_visual_proj_2(visual_embed)
                image_features_2 = image_features_2 / image_features_2.norm(dim=-1, keepdim=True) #(bs, nq, c)
                text_features_2 = self.text_embed_2
                text_features_2 = text_features_2 / text_features_2.norm(dim=-1, keepdim=True)    #(bs, c)
                logits_per_image_2 = logit_scale * torch.matmul(image_features_2, text_features_2.transpose(-1,-2))
                if self.ensemble_with_no_clip:
                    if self.debug:
                        import pdb; pdb.set_trace()
                    logits_no_clip = self.linear_classifier(visual_embed)
                    logits_per_image = (logits_per_image + logits_per_image_2 + logits_no_clip) / 3
                else:
                    logits_per_image = (logits_per_image + logits_per_image_2) / 2
            logits_list.append(logits_per_image)

        logits_list = torch.stack(logits_list)
        return logits_list

class DINO(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                    aux_loss=False, iter_update=False,
                    query_dim=2, 
                    random_refpoints_xy=False,
                    fix_refpoints_hw=-1,
                    num_feature_levels=1,
                    nheads=8,
                    # two stage
                    two_stage_type='no', # ['no', 'standard']
                    two_stage_add_query_num=0,
                    dec_pred_class_embed_share=True,
                    dec_pred_bbox_embed_share=True,
                    enc_cls_agn=False,
                    two_stage_class_embed_share=True,
                    two_stage_bbox_embed_share=True,
                    decoder_sa_type = 'sa',
                    num_patterns = 0,
                    dn_number = 100,
                    dn_box_noise_scale = 0.4,
                    dn_label_noise_ratio = 0.5,
                    dn_labelbook_size = 100,
                    dn_labelbook_reuse_cls = False,
                    use_cdn=True,
                    add_gt=False,
                    check_pos_dn = False,
                    matcher = None,
                    # clip
                    use_language=False,
                    # use_visual=False,
                    use_visual_distill=False,
                    use_mlp_proj=False,
                    use_cls_mlp_proj=True,
                    share_vl_proj=False,
                    distill_random_boxes=False,
                    use_clip_visual_query=False,
                    clip_visual_resolution=640,
                    use_cnn_clip=False,
                    clip_model='RN50',
                    distill_aux_layers=False,
                    two_stage_cls = False,
                    pre_compute_distill_target=False,
                    # imagenet pusedo labels
                    use_imagenet_pusedo_labels = False,
                    clip_pusedo_th = 0.05,
                    args = None,
                    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw: -1(default): learn w and h for each box seperately
                                >0 : given fixed number
                                -2 : learn a shared w and h
        """
        super().__init__()
        self.args = args
        self.cats = None # if cats is not None, extract text features
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.add_gt = add_gt
        self.dn_number = dn_number
        self.use_cdn = use_cdn
        self.check_pos_dn = check_pos_dn
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        # clip distill
        self.use_language = use_language
        self.use_visual = use_visual_distill or use_clip_visual_query or use_imagenet_pusedo_labels
        self.use_visual_distill = use_visual_distill
        self.distill_random_boxes = distill_random_boxes
        self.use_clip_visual_query = use_clip_visual_query
        self.use_cnn_clip = use_cnn_clip
        self.clip_visual_resolution = clip_visual_resolution
        self.distill_aux_layers= distill_aux_layers
        self.two_stage_cls = two_stage_cls and use_visual_distill
        self.pre_compute_distill_target = pre_compute_distill_target
        self.use_imagenet_pusedo_labels = use_imagenet_pusedo_labels
        self.clip_pusedo_th = clip_pusedo_th

        self.num_random_boxes = 100
        use_mlp_proj = use_mlp_proj and self.use_visual
        if not use_language:
            _class_embed = nn.Linear(hidden_dim, num_classes)
            _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        else :
            _class_embed = CLIPAlign(hidden_dim, True, use_cnn_clip=self.use_cnn_clip, clip_model=clip_model,
                                         use_label_enc=dn_labelbook_reuse_cls, use_visual=self.use_visual,
                                         use_mlp_cls=use_cls_mlp_proj and use_mlp_proj,
                                         use_mlp_distill=use_mlp_proj,
                                         share_vl_proj=share_vl_proj, use_clip_visual_proj=use_visual_distill,
                                         text_embed_type=args.text_embed_type, ensemble=args.clip_align_ensemble, ensemble_text_embed_type=args.text_embed_type_2,
                                         ensemble_with_no_clip=args.ensemble_with_no_clip,
                                         debug=args.debug)
            self.class_embed = _class_embed

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
        if not use_language:
            if dec_pred_class_embed_share:
                class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
            else:
                class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]
            self.class_embed = nn.ModuleList(class_embed_layerlist)
            self.transformer.decoder.class_embed = self.class_embed
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed

        self.matcher = matcher

        self.use_language_label_enc = False
        if dn_labelbook_reuse_cls :
            if self.use_language :
                self.label_enc = self.class_embed.get_label_enc
                self.use_language_label_enc = True
            else :
                self.label_enc = LabelEnc(self.class_embed[-1].weight)
        else :
            self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)
        self.transformer.label_enc = self.label_enc

        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
    
            if enc_cls_agn :
                # FIXME
                self.transformer.enc_out_class_embed = nn.Linear(hidden_dim, 1)
                self.transformer.enc_out_class_embed.bias.data = torch.ones(1) * bias_value
            elif two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            elif not self.use_language:
                self.transformer.enc_out_class_embed = nn.Linear(hidden_dim, num_classes)
                self.transformer.enc_out_class_embed.bias.data = torch.ones(num_classes) * bias_value
            else:
                self.transformer.enc_out_class_embed = CLIPAlign(hidden_dim, True,
                                                        use_label_enc=False,
                                                        use_visual=self.use_visual,
                                                        use_mlp_cls=use_cls_mlp_proj and use_mlp_proj,
                                                        use_mlp_distill=use_mlp_proj,
                                                        share_vl_proj=share_vl_proj, use_clip_visual_proj=use_visual_distill,
                                                        text_embed_type=args.text_embed_type, ensemble=args.clip_align_ensemble, ensemble_text_embed_type=args.text_embed_type_2,
                                                        ensemble_with_no_clip=args.ensemble_with_no_clip,
                                                        debug=args.debug)
                # if self.use_language_label_enc :
                #     del self.transformer.enc_out_class_embed.label_proj
    
            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        # self.replace_sa_with_double_ca = replace_sa_with_double_ca
        if decoder_sa_type == 'ca_label':
            self.label_embedding = nn.Embedding(num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

        if self.random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        
        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError('Unknown fix_refpoints_hw {}'.format(self.fix_refpoints_hw))

    @property
    def init_content_embedding(self):
        if self.transformer.tgt_embed is None :
            return None
        return self.transformer.tgt_embed.weight

    def _boxes2feature(self, boxes, boxes_scale, clip_features, text_embed):
        bs, nq = boxes.shape[:2]
        batch_ids = torch.arange(bs, device=boxes.device)[:,None,None].expand(-1, boxes.shape[1], -1)
        boxes_origin = box_ops.box_cxcywh_to_xyxy(boxes) * boxes_scale
        rois = torch.cat([batch_ids, boxes_origin], dim=-1)
        roi_features = self._get_roi(clip_features, rois.flatten(0,1), 1/self.class_embed.patch_size, self.class_embed.grid_size, is_image=False)
        boxes_feature = self.class_embed.clip.visual.attnpool(roi_features)
        boxes_feature = boxes_feature.view(bs, nq, *boxes_feature.shape[1:])
        boxes_feature = boxes_feature/ boxes_feature.norm(dim=-1, keepdim=True)
        boxes_logits = torch.matmul(boxes_feature, text_embed.transpose(-1,-2))
        boxes_logits = boxes_logits * self.class_embed.logit_scale.exp()
        return boxes_feature, boxes_logits

    def set_distill_outputs(self, extra_tgt, boxes, boxes_scale, pred_clip_hs, pred_clip_logits,out, dn_output):
        clip_features = self.clip_features
        text_embed = self.class_embed.text_embed
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        if boxes_scale.dim() == 2 :
            boxes_scale = boxes_scale[:, None]

        len_pred = out['pred_logits'].shape[1]
        if extra_tgt :
            boxes_feature, boxes_logits = self._boxes2feature(boxes, boxes_scale, clip_features, text_embed)
            len_dn = boxes_feature.shape[1]-len_pred
            boxes_feature_dn, boxes_feature = boxes_feature.split([len_dn, len_pred], dim=1)
            boxes_logits_dn, boxes_logits = boxes_logits.split([len_dn, len_pred], dim=1)
        else :
            boxes_feature_dn, boxes_feature, boxes_logits_dn, boxes_logits = [None] * 4

        pred_hs_dn, pred_hs = pred_clip_hs[:,:-len_pred], pred_clip_hs[:,-len_pred:]
        pred_clip_logits_dn, pred_clip_logits = pred_clip_logits[:,:-len_pred], pred_clip_logits[:,-len_pred:]
        if dn_output is not None :
            dn_output['pred_hs'] = pred_hs_dn
            dn_output['pred_clip_logits'] = pred_clip_logits_dn
            dn_output['hs_prompt'] = boxes_feature_dn
            dn_output['clip_logits'] = boxes_logits_dn

        out['pred_hs'] = pred_hs
        out['pred_clip_logits'] = pred_clip_logits
        out['hs_prompt'] = boxes_feature
        out['clip_logits'] = boxes_logits

    def clip_inference(self, samples, boxes, targets):
        clip_input_images = samples.tensors
        if self.use_cnn_clip :
            clip_input_images = self._denorm_images(clip_input_images)
            boxes_scale = torch.stack([t['size'][[1,0,1,0]] for t in targets]) #(bs, 4)
            boxes_scale = boxes_scale[:, None]
        else :
            boxes_scale = None
            batch_idx = torch.arange(clip_input_images.shape[0], device=clip_input_images.device).float()
            boxes = torch.stack([t['size'][[1,0,1,0]] for t in targets]) #(bs, 4)
            boxes[:,:2] = 0 #(0,0,w,h)

            rois = torch.cat([batch_idx[:,None], boxes], dim=-1)
            clip_input_images = self._get_roi(clip_input_images, rois)

        _, clip_features = self.class_embed.clip.encode_image(clip_input_images, ret_sp=True)
        bs, nq = boxes.shape[:2]
        if bs * nq == 0 :
            boxes = boxes.new_zeros(bs, 1, 4)
            bs, nq = boxes.shape[:2]

        batch_ids = torch.arange(bs, device=boxes.device)[:,None,None].expand(-1, boxes.shape[1], -1)
        boxes_origin = box_ops.box_cxcywh_to_xyxy(boxes) * boxes_scale
        rois = torch.cat([batch_ids, boxes_origin], dim=-1)
        aa = self._get_roi(clip_features, rois.flatten(0,1), 1/32, self.class_embed.grid_size, is_image=False)
        boxes_feature = self.class_embed.clip.visual.attnpool(aa)
        boxes_feature = boxes_feature.view(bs, nq, *boxes_feature.shape[1:])
        boxes_feature = boxes_feature/ boxes_feature.norm(dim=-1, keepdim=True)
        text_embed = self.class_embed.text_embed
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        boxes_logits = torch.matmul(boxes_feature, text_embed.transpose(-1,-2))
        boxes_logits = boxes_logits * self.class_embed.logit_scale.exp()

        out = {'pred_logits': boxes_logits, 'pred_boxes': boxes, 'dn_meta': None}
        return out

    def set_cats(self, cats):
        def set_total_text(model, value):
            if hasattr(model, 'set_total_text') :
                if getattr(self, 'text_embed', None) is None:
                    self.text_embed = model.set_total_text(value)
                else:
                    model.text_embed = self.text_embed

        total_text = cats
        self.apply(lambda x : set_total_text(x, total_text))

    def forward(self, samples: NestedTensor, targets:List=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # import pdb; pdb.set_trace()
        if self.use_language :
            self.class_embed.set_clip_mode() #freeze the norm layer in clip
        # do not change resolution
        # if self.use_visual :
        #     self.class_embed.clip.visual.change_input_resolution(self.clip_visual_resolution)
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        total_text = None

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        if self.use_visual and targets is not None:
            clip_input_images = samples.tensors
            if self.use_cnn_clip :
                clip_input_images = self._denorm_images(clip_input_images)
                boxes_scale = torch.stack([t['size'][[1,0,1,0]] for t in targets]) #(bs, 4)
                boxes_scale = boxes_scale[:, None]
            else :
                boxes_scale = None
                batch_idx = torch.arange(clip_input_images.shape[0], device=clip_input_images.device).float()
                boxes = torch.stack([t['size'][[1,0,1,0]] for t in targets]) #(bs, 4)
                boxes[:,:2] = 0 #(0,0,w,h)

                rois = torch.cat([batch_idx[:,None], boxes], dim=-1)
                clip_input_images = self._get_roi(clip_input_images, rois)
            _, clip_features = self.class_embed.clip.encode_image(clip_input_images, ret_sp=True)
            self.clip_features = clip_features

            # generate pusedo labels for imagenet
            if self.use_imagenet_pusedo_labels and targets[0].get('is_extra', False) :
                bbox_coord = torch.cat([t['boxes'] for t in targets])
                boxes_scale = torch.cat([t['size'][[1,0,1,0]][None].expand(len(t['boxes']),-1) for t in targets])
                bboxes_gt = boxes_scale *  box_ops.box_cxcywh_to_xyxy(bbox_coord)
                batch_idx = torch.arange(len(targets), device=clip_input_images.device, dtype=bbox_coord.dtype)
                batch_idx = torch.cat([b[None].expand(len(t['boxes'])) for b,t in zip(batch_idx, targets)])
                rois = torch.cat([batch_idx[:, None], bboxes_gt], dim=-1)
                text_embed = self.class_embed.text_embed
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

                if len(rois):
                    roi_features = self._get_roi(clip_features, rois, 1/self.class_embed.patch_size, self.class_embed.grid_size, is_image=False)
                    tgt_boxes_prompt = self.class_embed.clip.visual.attnpool(roi_features)
                    tgt_boxes_prompt = tgt_boxes_prompt/ tgt_boxes_prompt.norm(dim=-1, keepdim=True)
                else :
                    tgt_boxes_prompt = clip_features.new_zeros(0, text_embed.shape[-1])

                tgt_boxes_logits = tgt_boxes_prompt @ text_embed.t()
                tgt_boxes_logits = tgt_boxes_logits * self.class_embed.logit_scale.exp()
                target_len = [len(x['labels']) for x in targets]
                tgt_boxes_logits = tgt_boxes_logits.split(target_len, dim=0)
                for t, logits in zip(targets, tgt_boxes_logits):
                    probs = logits.softmax(-1)
                    if self.clip_pusedo_th > 0 :
                        filter_mask = (probs>self.clip_pusedo_th)
                    else :
                        if len(probs):
                            th = probs.max(-1, keepdim=True)[0]
                            if self.clip_pusedo_th > -1 :
                                th_low = abs(self.clip_pusedo_th)
                                th = th.clamp(min=th_low)
                        else :
                            th = 0.
                        filter_mask = (probs>=th)
                    filter_idx = filter_mask.nonzero()
                    # idx = (probs_topk > 0.05).nonzero()
                    origin_boxes = t['boxes']
                    t['boxes'] = origin_boxes[filter_idx[...,0]]
                    t['labels'] = filter_idx[...,1]

        if self.dn_number > 0 and targets is not None:
            input_query_label, input_query_bbox, attn_mask, dn_meta =\
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=self.training,num_queries=self.num_queries,num_classes=self.num_classes,
                                hidden_dim=self.hidden_dim,label_enc=self.label_enc,
                                init_content_query=self.transformer.tgt_embed,
                                use_cdn=self.use_cdn, check_pos_dn=self.check_pos_dn, add_gt=self.add_gt)

        elif self.add_gt_to_mask and targets is not None:
            input_query_label, input_query_bbox, attn_mask, dn_meta = prepare_for_cdn(
                            dn_args=(targets, 0, 0., 0.),
                            training=self.training,num_queries=self.num_queries,num_classes=self.num_classes,
                            hidden_dim=self.hidden_dim,label_enc=self.label_enc,
                            init_content_query=self.transformer.tgt_embed,
                            use_cdn=False, check_pos_dn=False, add_gt=False)

        else :
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        box2query=None
        if self.use_clip_visual_query:
            box2query = partial(self.box2clip_query,clip_features=clip_features)
        hs, reference, hs_enc, ref_enc, init_box_proposal, src_mems = self.transformer(
                            srcs, masks, input_query_bbox, poss,input_query_label,attn_mask,box2query=box2query)
        # In case num object=0
        if self.use_language_label_enc :
            hs[0] += self.class_embed.label_proj.weight[0,0] * 0.0
        else :
            hs[0]+=self.label_enc.weight[0,0]*0.0

        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        outputs_coord_list_reference = torch.stack(reference[:-1])

        # outputs_class = self.class_embed(hs)
        if not self.use_language :
            outputs_class = torch.stack([layer_cls_embed(layer_hs) for
                                        layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        else :
            outputs_class = self.class_embed.forward_hs(hs)
        if self.use_visual_distill :
            hs_stack = torch.stack(hs)
            pred_clip_hs = self.class_embed.proj_dino_hs(hs_stack)
            pred_clip_hs = pred_clip_hs/ pred_clip_hs.norm(dim=-1, keepdim=True)
            text_embed = self.class_embed.text_embed / self.class_embed.text_embed.norm(dim=-1, keepdim=True)
            pred_clip_logits = torch.matmul(pred_clip_hs, text_embed.transpose(-1,-2))
            pred_clip_logits = pred_clip_logits * self.class_embed.logit_scale.exp()
        # only two-stage prob during training
        if self.two_stage_cls and self.training :
            pred_clip_probs = pred_clip_logits.detach().softmax(-1)
            # TODO: how to add to class_logits
            pred_clip_probs_unsig = inverse_sigmoid(pred_clip_probs)
            outputs_class = outputs_class + pred_clip_probs_unsig
        if (self.dn_number > 0 or self.add_gt_to_mask) and dn_meta is not None:
            outputs_class, outputs_coord_list = \
                dn_post_process(outputs_class, outputs_coord_list,
                                dn_meta,self.aux_loss,self._set_aux_loss)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        if self.use_visual_distill and self.training :
            pred_clip_hs_this, pred_clip_logits_this = pred_clip_hs[-1], pred_clip_logits[-1]
            extra_box_clip_feature = not self.pre_compute_distill_target
            if self.pre_compute_distill_target :
                bbox_coord = torch.cat([t['boxes'] for t in targets])
                boxes_scale = torch.cat([t['size'][[1,0,1,0]][None].expand(len(t['boxes']),-1) for t in targets])
                bboxes_gt = boxes_scale *  box_ops.box_cxcywh_to_xyxy(bbox_coord)
                batch_idx = torch.arange(len(targets), device=clip_input_images.device, dtype=bbox_coord.dtype)
                batch_idx = torch.cat([b[None].expand(len(t['boxes'])) for b,t in zip(batch_idx, targets)])
                rois = torch.cat([batch_idx[:, None], bboxes_gt], dim=-1)
                if len(rois):
                    roi_features = self._get_roi(clip_features, rois, 1/self.class_embed.patch_size, self.class_embed.grid_size, is_image=False)
                    tgt_boxes_prompt = self.class_embed.clip.visual.attnpool(roi_features)
                    tgt_boxes_prompt = tgt_boxes_prompt/ tgt_boxes_prompt.norm(dim=-1, keepdim=True)
                else :
                    tgt_boxes_prompt = clip_features.new_zeros(0, text_embed.shape[-1])
                text_embed = self.class_embed.text_embed
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

                tgt_boxes_logits = tgt_boxes_prompt @ text_embed.t()
                tgt_boxes_logits = tgt_boxes_logits * self.class_embed.logit_scale.exp()
                target_len = [len(x['labels']) for x in targets]
                tgt_boxes_prompt_gather = tgt_boxes_prompt
                tgt_boxes_prompt = tgt_boxes_prompt.split(target_len, dim=0)
                tgt_boxes_logits = tgt_boxes_logits.split(target_len, dim=0)
                for t, clip_prompt, clip_logits in zip(targets, tgt_boxes_prompt, tgt_boxes_logits):
                    t['clip_prompt'] = clip_prompt
                    t['clip_logits'] = clip_logits

            # set distill pred(target) in output(dn_output) 
            bbox_coord = outputs_coord_list_reference[-1]
            dn_meta_distill = dn_meta.get('output_known_lbs_bboxes', None) if dn_meta else None
            boxes_scale = torch.stack([t['size'][[1,0,1,0]] for t in targets]) #(bs, 4)
            self.set_distill_outputs(extra_box_clip_feature, bbox_coord, boxes_scale, pred_clip_hs_this, pred_clip_logits_this, out, dn_meta_distill)

            if self.aux_loss and self.distill_aux_layers:
                for i, out_input in enumerate(out['aux_outputs']):
                    # hs_pred = hs[i]
                    pred_clip_hs_this, pred_clip_logits_this = pred_clip_hs[i], pred_clip_logits[i]
                    bbox_coord = outputs_coord_list_reference[i]
                    dn_meta_distill = dn_meta.get('output_known_lbs_bboxes', None) if dn_meta else None
                    if dn_meta_distill is not None :
                        dn_meta_distill = dn_meta_distill['aux_outputs'][i]
                    self.set_distill_outputs(extra_box_clip_feature, bbox_coord, boxes_scale, pred_clip_hs_this, pred_clip_logits_this, out_input, dn_meta_distill)

        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

            # prepare enc outputs
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out['enc_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]

        out['dn_meta'] = dn_meta

        if self.training :
            return out, targets
        return out

    def _box2clip_features(self, input_boxes, clip_features=None, boxes_scale=None):
        if clip_features is None :
            clip_features = self.clip_features
        input_boxes = box_ops.box_cxcywh_to_xyxy(input_boxes)
        input_boxes = input_boxes.clamp(min=0, max=1) # make sure the box in (0,1)
        bs, nq = input_boxes.shape[:2]
        batch_idx = torch.arange(bs, device=input_boxes.device)

        if boxes_scale is not None :
            input_boxes = input_boxes * (boxes_scale // self.class_embed.patch_size)
            spatial_scale = 1.
        else :
            spatial_scale = self.class_embed.grid_size
        query_rois = torch.cat([batch_idx[:,None,None].expand(-1, nq, -1), input_boxes], dim=-1)
        clip_query_features = self._get_roi(clip_features, query_rois.flatten(0,1), spatial_scale, output_size=1, is_image=False)
        clip_query_features = clip_query_features.squeeze(-1).squeeze(-1)
        return clip_query_features
    
    def box2clip_query(self, input_query_bbox, clip_features=None):
        '''
        input_query_bbox: (bs, nq, 4) normalized cxcywh
        clip_features: (bs, c, 7, 7) clip feaetures
        '''
        # if clip_features is None :
        #     clip_features = self.clip_features
        # input_query_bbox = box_ops.box_cxcywh_to_xyxy(input_query_bbox)
        # bs, nq = input_query_bbox.shape[:2]
        # batch_idx = torch.arange(bs, device=input_query_bbox.device)
        # query_rois = torch.cat([batch_idx[:,None,None].expand(-1, nq, -1), input_query_bbox], dim=-1)
        # clip_query_features = self._get_roi(clip_features, query_rois.flatten(0,1), 7, 1, is_image=False)
        bs, nq = input_query_bbox.shape[:2]
        clip_query_features = self._box2clip_features(input_query_bbox, clip_features)
        if nq == 0:
            clip_query_features = clip_query_features.new_zeros(bs, nq, clip_query_features.shape[-1])
        else :
            clip_query_features = clip_query_features.squeeze().view(bs, nq, -1)
        input_query_label = self.class_embed.label_proj(clip_query_features)
        return input_query_label

    def _denorm_images(self, images):
        de_norm = T.Compose([
            T.Normalize([0, 0, 0], [1/x for x in [0.229, 0.224, 0.225]]),
            T.Normalize([-x for x in [0.485, 0.456, 0.406]], [1, 1, 1]),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        images, _ = de_norm(images, None)
        return images

    def _get_roi(self, images, rois, spatial_scale=1., output_size=None, is_image=True):
        # rois = torch.cat([batch_idx[:,None], src_boxes], dim=1)
        if is_image :
            images = self._denorm_images(images)
            # de_norm = T.Compose([
            #     T.Normalize([0, 0, 0], [1/x for x in [0.229, 0.224, 0.225]]),
            #     T.Normalize([-x for x in [0.485, 0.456, 0.406]], [1, 1, 1]),
            #     T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            # ])
            # images, _ = de_norm(images, None)

        if output_size is None :
            output_size = self.class_embed.input_resolution
        obj_imgs = (ROIAlign(output_size, spatial_scale, 0, aligned=True).forward(images, rois).squeeze(1))
        return obj_imgs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses,
                 enc_cls_agn=False,
                 distill_type='l2',
                 distill_aux_layers=False,
                 use_dynamic_distill_weight=False,
                 clip_distill_objective='gt',
                 # federated loss
                 use_fed_loss=False, fed_num_sample_cats=50, use_fed_on_kd=False,
                 debug=False
                 ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.enc_cls_agn = enc_cls_agn
        self.distill_type = distill_type
        self.distill_aux_layers = distill_aux_layers
        self.clip_distill_objective = clip_distill_objective
        self.use_dynamic_distill_weight = use_dynamic_distill_weight
        self.cats = None

        self.use_fed_loss = use_fed_loss
        self.use_fed_on_kd = use_fed_on_kd and use_fed_loss
        self.fed_num_sample_cats = fed_num_sample_cats
        
        self.debug = debug

    def set_cats(self, cats):
        self.cats = cats
        if self.use_fed_loss:
            if self.debug:
                import pdb; pdb.set_trace()
            img_count_path = "./DATASET/v3det/cat_img_counts.npy"
            import os
            if os.path.exists(img_count_path):
                img_count = [0] + np.load(img_count_path).tolist()
                self.fed_weight = torch.tensor(img_count)
                self.fed_weight = self.fed_weight ** 0.5
            else:
                max_cid = max(self.cats)
                self.fed_weight = torch.tensor(
                    [self.cats.get(x, {'image_count':0}).get('image_count', 1) for x in range(max_cid+1)])
                self.fed_weight = self.fed_weight ** 0.5

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, enqueue_flag=False, is_enc_output=False):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        # if self.debug:
        #     import pdb; pdb.set_trace()
        # if is_enc_output and self.debug:
        #     import pdb; pdb.set_trace()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes if not is_enc_output else 1,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        if self.use_fed_loss and not is_enc_output:
            from .fed_loss import get_fed_loss_inds
            C = src_logits.shape[-1]
            fed_ids = get_fed_loss_inds(target_classes_o, self.fed_num_sample_cats, C, self.fed_weight)
            src_logits_fed = src_logits[...,fed_ids]
            target_classes_onehot_fed = target_classes_onehot[...,fed_ids]
            loss_ce = sigmoid_focal_loss(src_logits_fed, target_classes_onehot_fed, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        else :
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        # distill loss
        if 'pred_hs' in outputs :
            def get_dynamic_weight(tgt_logits):
                if self.use_dynamic_distill_weight:
                    tgt_prob = tgt_logits.softmax(-1)
                    tgt_entro = (-tgt_prob*tgt_prob.log()).sum(-1, keepdim=True)
                    weight = tgt_entro / math.log(tgt_logits.shape[-1]) * 2 #(normalize to (0,2))
                else :
                    weight = 1
                return weight

            if self.clip_distill_objective == 'gt':
                tgt_hs = torch.cat([t['clip_prompt'][i] for t, (_, i) in zip(targets, indices)], dim=0)
                tgt_logits = torch.cat([t['clip_logits'][i] for t, (_, i) in zip(targets, indices)], dim=0)
                if self.distill_type == 'clip_l1':
                    pred_hs = outputs['pred_hs']
                    pred_hs = pred_hs[idx]
                    pred_hs = pred_hs / pred_hs.norm(dim=-1, keepdim=True)
                    losses['loss_distill'] = F.l1_loss(pred_hs, tgt_hs, reduction='sum') / num_boxes
                elif self.distill_type == 'clip_logits':
                    pred_logits = outputs['pred_clip_logits']
                    pred_logits = pred_logits[idx]
                    distill_weight = get_dynamic_weight(tgt_logits)
                    if self.use_fed_on_kd :
                        pred_logits, tgt_logits = pred_logits[..., fed_ids], tgt_logits[..., fed_ids]
                    losses['loss_distill'] = (F.kl_div(pred_logits.log_softmax(-1), tgt_logits.softmax(-1), reduction='none')*distill_weight).sum()/ num_boxes
                    # losses['loss_distill'] = F.kl_div(pred_logits.log_softmax(-1), tgt_logits.softmax(-1), reduction='sum') / num_boxes
                else :
                    raise NotImplementedError
            elif self.clip_distill_objective in ('pred', 'pred_all'):
                if self.distill_type == 'clip_l1':
                    pred_hs = outputs['pred_hs']
                    tgt_hs = outputs['hs_prompt']
                    pred_hs = pred_hs / pred_hs.norm(dim=-1, keepdim=True)
                    tgt_hs = tgt_hs / tgt_hs.norm(dim=-1, keepdim=True)
                    if self.clip_distill_objective == 'pred':
                        pred_hs, tgt_hs = pred_hs[idx], tgt_hs[idx]
                        losses['loss_distill'] = F.l1_loss(pred_hs, tgt_hs, reduction='sum') / num_boxes
                    else :
                        bs, nq = pred_hs.shape[:2]
                        losses['loss_distill'] = F.l1_loss(pred_hs, tgt_hs, reduction='sum') / (bs*nq)
                elif self.distill_type == 'clip_logits':
                    pred_logits = outputs['pred_clip_logits']
                    tgt_logits = outputs['clip_logits']
                    if self.clip_distill_objective == 'pred':
                        pred_logits, tgt_logits = pred_logits[idx], tgt_logits[idx]
                        distill_weight = get_dynamic_weight(tgt_logits)
                        if self.use_fed_on_kd :
                            pred_logits, tgt_logits = pred_logits[..., fed_ids], tgt_logits[..., fed_ids]
                        losses['loss_distill'] = (F.kl_div(pred_logits.log_softmax(-1), tgt_logits.softmax(-1), reduction='none')*distill_weight).sum() / num_boxes
                    else :
                        bs, nq = pred_logits.shape[:2]
                        distill_weight = get_dynamic_weight(tgt_logits)
                        if self.use_fed_on_kd :
                            pred_logits, tgt_logits = pred_logits[..., fed_ids], tgt_logits[..., fed_ids]
                        losses['loss_distill'] = (F.kl_div(pred_logits.log_softmax(-1), tgt_logits.softmax(-1), reduction='none')*distill_weight).sum() / (bs*nq)
                else :
                    raise NotImplementedError

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        losses = {}
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes


        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets)

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t[1]) for t in indices)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta']

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.range(0, len(targets[i]['labels']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))


            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False, 'enqueue_flag': True}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            if self.distill_type == 'clip_l1' or self.distill_type == 'clip_logits':
                l_dict['loss_distill_dn'] = torch.as_tensor(0.).to('cuda')

            losses.update(l_dict)

        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                                 **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict['loss_bbox_dn']=torch.as_tensor(0.).to('cuda')
                    l_dict['loss_giou_dn']=torch.as_tensor(0.).to('cuda')
                    l_dict['loss_ce_dn']=torch.as_tensor(0.).to('cuda')
                    l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                    if self.distill_aux_layers and (self.distill_type == 'clip_l1' or self.distill_type == 'clip_logits'):
                        l_dict['loss_distill_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)


        # interm_outputs loss
        if 'interm_outputs' in outputs:
            if self.debug:
                import pdb; pdb.set_trace()
            interm_outputs = outputs['interm_outputs']
            if self.enc_cls_agn:
                enc_targets = copy.deepcopy(targets)
                for enc_t in enc_targets :
                    enc_t['labels'] = torch.zeros_like(enc_t['labels'])
            else :
                enc_targets = targets
            indices = self.matcher(interm_outputs, enc_targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks' :
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                    if self.enc_cls_agn:
                        kwargs['is_enc_output'] = True
                l_dict = self.get_loss(loss, interm_outputs, enc_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            if self.enc_cls_agn:
                enc_targets = copy.deepcopy(targets)
                for enc_t in enc_targets :
                    enc_t['labels'] = torch.zeros_like(enc_t['labels'])
            else :
                enc_targets = targets
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, enc_targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, enc_outputs, enc_targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups

class PostProcessTTA(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100, nms_iou_threshold=-1, matcher=None) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold
        self.matcher = matcher

    @torch.no_grad()
    def forward(self, outputs, target_sizes, catids=None, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:            
            # item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]
            # results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]

            # per-class nms
            scores_all = []
            labels_all = []
            boxes_all = []        
            labels_set = set(labels.view(-1).tolist())
            for label in labels_set:
                idx = labels==label
                labels_indexed = labels[idx]
                boxes_indexed = boxes[idx]
                scores_indexed = scores[idx]
                item_indices = nms(boxes_indexed, scores_indexed, iou_threshold=self.nms_iou_threshold)
                scores_all.append(scores_indexed[item_indices])            
                labels_all.append(labels_indexed[item_indices])
                boxes_all.append(boxes_indexed[item_indices])
            # keep top predictions
            _, topk_inds = torch.topk(torch.cat(scores_all), min(300, torch.cat(scores_all).shape[0]))

            if test:
                boxes_all = torch.cat(boxes_all)
                assert not not_to_xyxy
                boxes_all[:,2:] = boxes_all[:,2:] - boxes_all[:,:2]
            else:
                boxes_all = torch.cat(boxes_all)

            results = [{'scores': torch.cat(scores_all)[topk_inds], 'labels': torch.cat(labels_all)[topk_inds], 'boxes': boxes_all[topk_inds]}]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100, nms_iou_threshold=-1, use_opt=False) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold
        self.use_opt = use_opt

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        if 'pred_masks' in outputs :
            bs, nq_topk = topk_boxes.shape
            outputs['pred_masks'] = torch.gather(outputs['pred_masks'], 1, topk_boxes.view(
                            bs, nq_topk, 1, 1, 1).expand(-1, -1, -1,*outputs['pred_masks'].shape[-2:]))
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.use_opt :
            item_indices = [batched_nms(b, s, l, 0.7) for b,s,l in zip(boxes, scores, labels)]
            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
            outputs['item_indices'] = item_indices
        elif self.nms_iou_threshold > 0:
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]
            # import ipdb; ipdb.set_trace()
            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

@MODULE_BUILD_FUNCS.registe_with_name(module_name='richsem')
def build_richsem(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     # for panoptic, we just add a num_classes that is large enough to hold
    #     # max_obj_id + 1, but the exact value doesn't really matter
    #     num_classes = 250
    # if args.dataset_file == 'o365':
    #     num_classes = 366
    # if args.dataset_file == 'vanke':
    #     num_classes = 51
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deformable_transformer(args)

    try:
        dn_labelbook_size = args.dn_labelbook_size
    except:
        dn_labelbook_size = num_classes

    try:
        dec_pred_class_embed_share = args.dec_pred_class_embed_share
    except:
        dec_pred_class_embed_share = True
    try:
        dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    except:
        dec_pred_bbox_embed_share = True
    try :
        use_cdn = args.use_cdn
    except :
        use_cdn = True

    matcher = build_matcher(args)

    model = DINO(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # matcher
        matcher=matcher,
        # two stage
        two_stage_type=args.two_stage_type,
        enc_cls_agn=args.enc_cls_agn,
        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        decoder_sa_type=args.decoder_sa_type,
        num_patterns=args.num_patterns,
        dn_number = args.dn_number if args.use_dn else 0,
        dn_box_noise_scale = args.dn_box_noise_scale,
        dn_label_noise_ratio = args.dn_label_noise_ratio,
        dn_labelbook_size = dn_labelbook_size,
        dn_labelbook_reuse_cls = args.dn_labelbook_reuse_cls,
        use_cdn=use_cdn, check_pos_dn=args.check_pos_dn,
        add_gt=args.add_gt,
        use_language=args.use_language,
        use_visual_distill=args.use_visual_distill,
        use_mlp_proj=args.use_mlp_proj,
        use_cls_mlp_proj=args.use_cls_mlp_proj,
        share_vl_proj=args.share_vl_proj,
        distill_random_boxes=args.distill_random_boxes,
        use_clip_visual_query=args.use_clip_visual_query,
        clip_visual_resolution=args.clip_visual_resolution,
        distill_aux_layers=args.distill_aux_layers,
        use_cnn_clip=args.use_cnn_clip,
        clip_model=args.clip_model,
        two_stage_cls=args.two_stage_cls,
        pre_compute_distill_target=args.clip_distill_objective in ('gt',),
        # pusedo labels
        use_imagenet_pusedo_labels=args.use_imagenet_pusedo_labels,
        clip_pusedo_th=args.clip_pusedo_th,
        args=args,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    
    # for DN training
    if args.use_dn:
        weight_dict['loss_ce_dn'] = args.cls_loss_coef
        weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_giou_dn'] = args.giou_loss_coef
        if args.add_gt or args.use_visual_distill :
            weight_dict['loss_distill_dn'] = args.distill_loss_coef

    if args.add_gt or args.use_visual_distill :
        weight_dict["loss_distill"] = args.distill_loss_coef

    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses,
                             enc_cls_agn=args.enc_cls_agn,
                             distill_type=args.distill_type,
                             distill_aux_layers=args.distill_aux_layers,
                             clip_distill_objective=args.clip_distill_objective,
                             use_dynamic_distill_weight=args.use_dynamic_distill_weight,
                             use_fed_loss=args.use_fed_loss,
                             fed_num_sample_cats=args.fed_num_sample_cats, use_fed_on_kd=args.use_fed_on_kd,
                             debug=args.debug,
                             )
    criterion.to(device)
    # postprocessors = {'bbox': PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold,
    #                                       use_opt=args.matcher_type=='OptMatcher'),}
    matcher4tta = build_matcher4tta(args)
    postprocessors = {
        'bbox': PostProcessTTA(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold, matcher=matcher4tta) if args.use_tta else \
            PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold)
    }

    if args.masks :
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors