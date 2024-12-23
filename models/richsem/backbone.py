from collections import OrderedDict
import os

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List


from util.misc import NestedTensor, clean_state_dict, is_main_process

from .position_encoding import build_position_encoding
from .convnext import build_convnext
from .swin_transformer import build_swin_transformer
from .focal import build_focalnet


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_indices: list,
                 total_finetune=False):
        super().__init__()
        if not total_finetune :
            for name, parameter in backbone.named_parameters():
                if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                    parameter.requires_grad_(False)

        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update({"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)})

        # if len:
        #     if use_stage1_feature:
        #         return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        #     else:
        #         return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        # else:
        #     return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        # import ipdb; ipdb.set_trace()
        return out


from timm import create_model
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model
from timm.models.resnet import ResNet, Bottleneck
from timm.models.resnet import default_cfgs as default_cfgs_resnet
import copy

model_params = {
    'resnet50_in21k': dict(block=Bottleneck, layers=[3, 4, 6, 3]),
}

def create_timm_resnet_21k(pretrained=False, **kwargs):
    variant = 'resnet50_in21k'
    params = model_params[variant]
    default_cfgs_resnet['resnet50_in21k'] = \
        copy.deepcopy(default_cfgs_resnet['resnet50'])
    default_cfgs_resnet['resnet50_in21k']['url'] = \
        'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth'
    default_cfgs_resnet['resnet50_in21k']['num_classes'] = 11221

    return build_model_with_cfg(
        ResNet, variant, pretrained,
        default_cfg=default_cfgs_resnet[variant],
        pretrained_custom_load=True,
        **params,
        **kwargs)


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 dilation: bool,
                 return_interm_indices:list,
                 batch_norm=FrozenBatchNorm2d,
                 args=None
                 ):
        total_finetune = False
        if name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
            if args.resnet_pretrain_path :
                backbone = create_timm_resnet_21k(pretrained=False,
                                                  norm_layer=batch_norm
                                                  )
                pretrainedpath = args.resnet_pretrain_path
                checkpoint = torch.load(pretrainedpath, map_location='cpu')['model']
                _keys = backbone.load_state_dict(checkpoint)
                total_finetune = True
                print(_keys)
            else :
                backbone = getattr(torchvision.models, name)(
                    replace_stride_with_dilation=[False, False, dilation],
                    pretrained=is_main_process(), norm_layer=batch_norm)
        elif name in ['clip_rn50']:
            import clip
            backbone = clip.load('RN50', 'cpu', False)[0].visual.float()
        else:
            raise NotImplementedError("Why you can get here with name {}".format(name))
        # num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        assert name not in ('resnet18', 'resnet34'), "Only resnet50 and resnet101 are available."
        assert return_interm_indices in [[0,1,2,3], [1,2,3], [3]]
        num_channels_all = [256, 512, 1024, 2048]
        num_channels = num_channels_all[4-len(return_interm_indices):]
        super().__init__(backbone, train_backbone, num_channels, return_interm_indices, 
                            total_finetune=total_finetune)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone: 
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords: 
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0,1,2,3], [1,2,3], [3]]
    backbone_freeze_keywords = args.backbone_freeze_keywords
    use_checkpoint = getattr(args, 'use_checkpoint', False)

    if args.backbone in ['resnet50', 'resnet101', 'clip_rn50']:
        backbone = Backbone(args.backbone, train_backbone, args.dilation,   
                                return_interm_indices,   
                                batch_norm=FrozenBatchNorm2d,
                                args=args)
        bb_num_channels = backbone.num_channels

    elif args.backbone in ['swin_T_224_1k', 'swin_B_224_22k', 'swin_B_224_1k', 'swin_B_384_22k', 'swin_L_224_22k', 'swin_L_384_22k']:
        pretrain_img_size = int(args.backbone.split('_')[-2])
        backbone = build_swin_transformer(args.backbone, \
                    pretrain_img_size=pretrain_img_size, \
                    out_indices=tuple(return_interm_indices), \
                dilation=args.dilation, use_checkpoint=use_checkpoint)

        # freeze some layers
        if backbone_freeze_keywords is not None:
            for name, parameter in backbone.named_parameters():
                for keyword in backbone_freeze_keywords:
                    if keyword in name:
                        parameter.requires_grad_(False)
                        break

        pretrained_dir = args.backbone_dir
        PTDICT = {
            'swin_T_224_1k': 'swin_tiny_patch4_window7_224.pth',
            'swin_B_384_22k': 'swin_base_patch4_window12_384_22k.pth',
            'swin_L_384_22k': 'swin_large_patch4_window12_384_22k.pth',
            'swin_B_224_1k': 'swin_base_patch4_window7_224.pth',
        }
        pretrainedpath = os.path.join(pretrained_dir, PTDICT[args.backbone])
        checkpoint = torch.load(pretrainedpath, map_location='cpu')['model']
        from collections import OrderedDict
        def key_select_function(keyname):
            if 'head' in keyname:
                return False
            if args.dilation and 'layers.3' in keyname:
                return False
            return True
        _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if key_select_function(k)})
        _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)
        print(str(_tmp_st_output))
        bb_num_channels = backbone.num_features[4 - len(return_interm_indices):]
    elif args.backbone in ['focalnet_L_384_22k', 'focalnet_L_384_22k_fl4', 'focalnet_XL_384_22k', 'focalnet_H_224_22k_fl4_fd']:
        # added by Jianwei
        backbone = build_focalnet(args.backbone, \
                    # focal_levels=args.focal_levels, \
                    # focal_windows=args.focal_windows, \
                    out_indices=tuple(return_interm_indices), \
                    use_checkpoint=use_checkpoint)

        # freeze some layers
        if backbone_freeze_keywords is not None:
            for name, parameter in backbone.named_parameters():
                for keyword in backbone_freeze_keywords:
                    if keyword in name:
                        parameter.requires_grad_(False)
                        break

        pretrained_dir = args.backbone_dir
        PTDICT = {
            'focalnet_L_384_22k': 'focalnet_large_lrf_384.pth',
            'focalnet_L_384_22k_fl4': 'focalnet_large_lrf_384_fl4.pth',
            'focalnet_XL_384_22k': 'focalnet_xlarge_lrf_384.pth',
            'focalnet_huge_224_22k': 'focalnet_huge_lrf_224.pth', 
            'focalnet_H_224_22k_fl4_fd': 'focalnet_huge_lrf_224_fl4.pth'
        }
        try :
            pretrainedpath = os.path.join(pretrained_dir, PTDICT[args.backbone])
            checkpoint = torch.load(pretrainedpath, map_location='cpu')['model']
            from collections import OrderedDict
            def key_select_function(keyname):
                if 'head' in keyname:
                    return False
                if args.dilation and 'layers.3' in keyname:
                    return False
                return True        
            _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if key_select_function(k)})

            # For using larger kernel size during finetuning
            for name, parameter in backbone.named_parameters():
                if "modulation.focal_layers" in name:
                    kernel_size = parameter.shape[2:]
                    if _tmp_st[name].shape[2:] != kernel_size:
                        assert kernel_size[0] > _tmp_st[name].shape[2]
                        assert kernel_size[1] > _tmp_st[name].shape[3]

                        offset_h = (kernel_size[0] - _tmp_st[name].shape[2]) // 2
                        offset_w = (kernel_size[1] - _tmp_st[name].shape[3]) // 2

                        new_params = parameter.clone().fill_(0.0)
                        new_params[:,:,offset_h:offset_h+_tmp_st[name].shape[2],offset_w:offset_w+_tmp_st[name].shape[3]] = _tmp_st[name]
                        _tmp_st[name] = new_params

            _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)
            print(str(_tmp_st_output))
        except:
            print('no pretrained ckpt found')
        bb_num_channels = backbone.num_features[4 - len(return_interm_indices):]    
    elif args.backbone in ['convnext_xlarge_22k']:
        backbone = build_convnext(modelname=args.backbone, pretrained=True, out_indices=tuple(return_interm_indices),backbone_dir=args.backbone_dir)
        bb_num_channels = backbone.dims[4 - len(return_interm_indices):]
    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))
    

    assert len(bb_num_channels) == len(return_interm_indices), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"


    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels 
    assert isinstance(bb_num_channels, List), "bb_num_channels is expected to be a List but {}".format(type(bb_num_channels))
    return model