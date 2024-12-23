import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import MLP
from detectron2.layers.batch_norm import get_norm
from detectron2.layers import Conv2d

class CustomLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 0.00001, elementwise_affine: bool = True) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, x) :
        if x.dim() == 4 :
            # bs, c, h, w
            bs, c, h, w = x.shape
            x = x.flatten(2).transpose(1,2) #(bs, nq, c)
            x = super().forward(x)
            x = x.transpose(1,2).view(bs, c, h, w).contiguous()
            return x
        return super().forward(x)

def conv_with_kaiming_uniform(
        norm=None, activation=None,
        use_deformable=False, use_sep=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        if use_deformable:
            conv_func = DFConv2d
        else:
            conv_func = Conv2d
        if use_sep:
            assert in_channels == out_channels
            groups = in_channels
        else:
            groups = 1
        conv = conv_func(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            groups=groups,
            bias=(norm is None)
        )
        if not use_deformable:
            # Caffe2 implementation uses XavierFill, which in fact
            # corresponds to kaiming_uniform_ in PyTorch
            nn.init.kaiming_uniform_(conv.weight, a=1)
            if norm is None:
                nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if norm is not None and len(norm) > 0:
            if norm== 'LN':
                norm_module = CustomLayerNorm(out_channels)
            elif norm == "GN":
                norm_module = nn.GroupNorm(32, out_channels)
            else:
                norm_module = get_norm(norm, out_channels)
            module.append(norm_module)
        if activation is not None:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv

class MaskConvBranch(nn.Module):
    def __init__(self, in_channel, out_channels, 
                 num_conv=4, hidden_channels=128, out_stride=8):
        super().__init__()
        self.sem_loss_on = False
        self.num_outputs = out_channels
        norm = 'LN'
        num_convs = num_conv
        channels = hidden_channels
        self.out_stride = out_stride

        self.in_features = [0, 1, 2]
        feature_channels = [in_channel] * 3

        conv_block = conv_with_kaiming_uniform(norm, activation=True)

        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(conv_block(
                feature_channels[in_feature],
                channels, 3, 1
            ))

        tower = []
        for i in range(num_convs):
            tower.append(conv_block(
                channels, channels, 3, 1
            ))
        tower.append(nn.Conv2d(
            channels, max(self.num_outputs, 1), 1
        ))
        self.add_module('tower', nn.Sequential(*tower))

    def forward(self, features):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])

                target_h, target_w = x.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p

        mask_feats = self.tower(x)

        if self.num_outputs == 0:
            mask_feats = mask_feats[:, :self.num_outputs]

        return mask_feats


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim, dim_div=32):
        super().__init__()

        # inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        inter_dims = [dim, context_dim, context_dim, context_dim, context_dim, context_dim]

        # used after upsampling to reduce dimention of fused features!
        self.lay1 = torch.nn.Conv2d(dim, dim//4, 3, padding=1)
        self.lay2 = torch.nn.Conv2d(dim//4, dim//dim_div, 3, padding=1)
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.dcn = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.dim = dim

        if fpn_dims != None:
            self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
            self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
            self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for name, m in self.named_modules():
            if name == "conv_offset":
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            else:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    nn.init.constant_(m.bias, 0)


    def forward(self, x, fpns=None):

        if fpns != None:
            cur_fpn = self.adapter1(fpns[0])
            if cur_fpn.size(0) != x[-1].size(0):
                cur_fpn = _expand(cur_fpn, x[-1].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-1]) / 2
        else:
            fused_x = x[-1]
        fused_x = self.lay3(fused_x)
        fused_x = F.relu(fused_x)

        if fpns != None:
            cur_fpn = self.adapter2(fpns[1])
            if cur_fpn.size(0) != x[-2].size(0):
                cur_fpn = _expand(cur_fpn, x[-2].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-2]) / 2 + F.interpolate(fused_x, size=cur_fpn.shape[-2:], mode="nearest")
        else:
            fused_x = x[-2] + F.interpolate(fused_x, size=x[-2].shape[-2:], mode="nearest")
        fused_x = self.lay4(fused_x)
        fused_x = F.relu(fused_x)

        if fpns != None:
            cur_fpn = self.adapter3(fpns[2])
            if cur_fpn.size(0) != x[-3].size(0):
                cur_fpn = _expand(cur_fpn, x[-3].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-3]) / 2 + F.interpolate(fused_x, size=cur_fpn.shape[-2:], mode="nearest")
        else:
            fused_x = x[-3] + F.interpolate(fused_x, size=x[-3].shape[-2:], mode="nearest")
        fused_x = self.dcn(fused_x)
        fused_x = F.relu(fused_x)
        fused_x = self.lay1(fused_x)
        fused_x = F.relu(fused_x)
        fused_x = self.lay2(fused_x)
        fused_x = F.relu(fused_x)

        return fused_x


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MaskBranch(nn.Module):
    def __init__(self, hidden_dim, 
                    rel_coord=True, use_relative_hw=False,
                    channel_div=32, dy_mask_channel=8, is_concat=True) -> None:
        super().__init__()
        self.rel_coord = rel_coord
        self.is_concat = is_concat
        self.use_relative_hw = use_relative_hw

        self.in_channels = hidden_dim // channel_div
        if is_concat :
            self.in_channels = self.in_channels * 2
        self.dynamic_mask_channels = dy_mask_channel
        self.controller_layers = 3
        self.mask_out_stride = 4
        self.up_rate = 8 // self.mask_out_stride

        weight_nums, bias_nums = [], []
        for l in range(self.controller_layers):
            if l == 0:
                if self.rel_coord:
                    weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            elif l == self.controller_layers - 1:
                weight_nums.append(self.dynamic_mask_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)


        self.controller = MLP(hidden_dim, hidden_dim, self.num_gen_params, 3)
        # for concating backbone and encoder
        if is_concat :
            mask_dim = hidden_dim*2
        else :
            mask_dim = hidden_dim
        self.mask_head = MaskHeadSmallConv(mask_dim, None, mask_dim, dim_div=channel_div)
        # self.mask_head = MaskConvBranch(mask_dim, mask_dim// channel_div)


    def forward(self, hs, memory, references, indices_list, targets):
        outputs_masks = []
        output_params = []

        for lvl, indices in enumerate(indices_list):
            reference_points, mask_head_params, num_insts = [], [], []
            reference = references[lvl]

            dynamic_mask_head_params = self.controller(hs[lvl])
            if self.training :
                for i, indice in enumerate(indices):
                    img_size = targets[i]['size'][[1,0,1,0]]
                    ref_cur_f = reference[i]
                    ref_cur_f = ref_cur_f * img_size
                    pred_i, tgt_j = indice
                    reference_points.append(ref_cur_f[pred_i].unsqueeze(0))
                    num_insts.append(pred_i.numel())
                    mask_head_params.append(dynamic_mask_head_params[i, pred_i].unsqueeze(0))
            else :
                img_sizes = [t['size'][[1,0,1,0]] for t in targets]
                reference_points = [(x * size).unsqueeze(0) for x, size in zip(reference, img_sizes)]
                num_insts = [len(x) for x in dynamic_mask_head_params]
                mask_head_params = [x.unsqueeze(0) for x in dynamic_mask_head_params]

            reference_points = torch.cat(reference_points, dim=1)
            mask_head_params = torch.cat(mask_head_params, dim=1)

            pred_masks = self.forward_mask_head_train(memory, reference_points, mask_head_params, num_insts)
            if self.training :
                pred_masks = torch.cat(pred_masks)
            else :
                pred_masks = torch.stack(pred_masks) #(bs, nq, 1, h, w)

            output_params.append(dynamic_mask_head_params)
            outputs_masks.append(pred_masks)

        # outputs_masks = [torch.cat(x) for x in outputs_masks]
        return outputs_masks, output_params


    def forward_mask_head_train(self, src_feats, reference_points, mask_head_params, num_insts):
        bs, c = src_feats[0].shape[:2]

        # encod_feat_l: num_layers x [bs, C, num_frames, hi, wi]
        encod_feat_l = src_feats
        spatial_indx = 0
        # for feat_l in range(self.detr.num_feature_levels - 1):
        #     h, w = spatial_shapes[feat_l]
        #     mem_l = feats[:, spatial_indx: spatial_indx + 1 * h * w, :].reshape(bs, 1, h, w, c).permute(0,4,1,2,3)
        #     encod_feat_l.append(mem_l)
        #     spatial_indx += 1 * h * w
        
        pred_masks = []
        # for iframe in range(1):
        #     encod_feat_f = []
        #     for lvl in range(self.detr.num_feature_levels - 1):
        #         encod_feat_f.append(encod_feat_l[lvl][:, :, iframe, :, :]) # [bs, C, hi, wi]
            # feats = [] # features[3], features[2], features[1]
            # for i in range(self.detr.num_feature_levels - 1, 0, -1):
            #     N, _c, _h, _w = features[i].tensors.shape
            #     feats.append(features[i].tensors.reshape(bs, self.detr.num_frames, _c, _h, _w)[:,iframe,:,:,:])
            
        if 1 :
            encod_feat_f = src_feats
            decod_feat_f = self.mask_head(encod_feat_f)
          
            ######### conv ##########
            mask_logits = self.dynamic_mask_with_coords(decod_feat_f, reference_points, mask_head_params, 
                                                        num_insts=num_insts,
                                                        mask_feat_stride=8,
                                                        rel_coord=self.rel_coord )
            # mask_logits: [1, num_queries_all, H/4, W/4]

            # mask_f = mask_logits.unsqueeze(2).reshape(bs, nq, 1, decod_feat_f.shape[-2], decod_feat_f.shape[-1])  # [bs, selected_queries, 1, H/4, W/4]
            mask_f = []
            inst_st = 0
            for num_inst in num_insts:
                # [1, selected_queries, 1, H/4, W/4]
                mask_f.append(mask_logits[:, inst_st: inst_st + num_inst, :, :].unsqueeze(2))
                inst_st += num_inst

        
        output_pred_masks = [x.squeeze(0) for x in mask_f]
        return output_pred_masks

    def dynamic_mask_with_coords(self, mask_feats, reference_points, mask_head_params, num_insts, 
                                 mask_feat_stride, rel_coord=True):
        # mask_feats: [N, C/32, H/8, W/8]
        # reference_points: [1, \sum{selected_insts}, 2]
        # mask_head_params: [1, \sum{selected_insts}, num_params]
        # return:
        #     mask_logits: [1, \sum{num_queries}, H/8, W/8]
        device = mask_feats.device

        N, in_channels, H, W = mask_feats.size()
        num_insts_all = reference_points.shape[1]

        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3), 
            device=device, stride=mask_feat_stride)
        # locations: [H*W, 2]
        
        if rel_coord:
            # instance_locations = reference_points
            instance_locations, instance_whs = reference_points.split([2,2], dim=-1)
            relative_coords = instance_locations.reshape(1, num_insts_all, 1, 1, 2) - locations.reshape(1, 1, H, W, 2)
            if self.use_relative_hw :
                instance_hws = instance_whs[...,[1,0]]
                relative_coords = 1 - (relative_coords / instance_hws.view(1, num_insts_all, 1, 1, 2) * 2).abs()
                relative_coords = 1 - relative_coords.abs()
                relative_coords = (relative_coords / instance_whs.view(1, num_insts_all, 1, 1, 2) * 2)

            relative_coords = relative_coords.float()
            relative_coords = relative_coords.permute(0, 1, 4, 2, 3).flatten(-2, -1)
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                # [1, num_queries * (C/32+2), H/8 * W/8]
                relative_coords_b = relative_coords[:, inst_st: inst_st + num_inst, :, :]
                mask_feats_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = torch.cat([relative_coords_b, mask_feats_b], dim=2)

                mask_head_inputs.append(mask_head_b)
                inst_st += num_inst

        else:
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                mask_head_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = mask_head_b.reshape(1, -1, H, W)
                mask_head_inputs.append(mask_head_b)
        
        # mask_head_inputs: [1, \sum{num_queries * (C/32+2)}, H/8, W/8]
        mask_head_inputs = torch.cat(mask_head_inputs, dim=1)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        # mask_head_params: [num_insts_all, num_params]
        mask_head_params = torch.flatten(mask_head_params, 0, 1)
       
        if num_insts_all != 0:
            weights, biases = parse_dynamic_params(
                mask_head_params, self.dynamic_mask_channels,
                self.weight_nums, self.bias_nums
            )

            mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, mask_head_params.shape[0])
        else:
            mask_logits = mask_head_inputs
            return mask_logits
        # mask_logits: [1, num_insts_all, H/8, W/8]
        mask_logits = mask_logits.reshape(-1, 1, H, W)

        # upsample predicted masks
        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0

        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        mask_logits = mask_logits.reshape(1, -1, mask_logits.shape[-2], mask_logits.shape[-1])
        # mask_logits: [1, num_insts_all, H/4, W/4]

        return mask_logits


    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x



def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]

def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits

