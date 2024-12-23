import torch
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from util import box_ops
import torch.nn.functional as F
import random
from util.box_ops import box_cxcywh_to_xyxy, box_iou

def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc,
                    init_content_query=None, use_cdn=False, check_pos_dn=False, add_gt=False):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        if add_gt :
            dn_number += 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        num_pos_per_group = len(labels)
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            if add_gt :
                p[:num_pos_per_group] = 1
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)

        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            if add_gt :
                rand_part[:num_pos_per_group] = 0
            known_bbox_pre = known_bbox_.clone()
            known_bbox_ = known_bbox_pre + torch.mul(rand_part,
                                                  diff).cuda() * box_noise_scale
            if check_pos_dn and len(boxes):
                num_try = 5
                pos_bbox_pre = known_bbox_pre[positive_idx]
                pos_bbox_iou_label = torch.arange(len(boxes), device=boxes.device).repeat(dn_number)
                need_check = torch.ones(len(positive_idx), dtype=torch.bool)
                pos_rand_part = rand_part[positive_idx]
                pos_diff = diff[positive_idx]
                valid_mask = torch.zeros(len(boxes), len(boxes), dtype=torch.bool)
                batch_num = [len(t['boxes']) for t in targets]
                count = 0
                for n in batch_num :
                    valid_mask[count:count+n, count:count+n] = True
                    count += n
                valid_mask = valid_mask.repeat(dn_number, 1)
                for i in range(num_try):
                    # # known_bbox_expand = known_bbox_expand[:, positive_idx]
                    # known_bbox_expand_pos = known_bbox_expand[positive_idx]
                    # ious = box_iou(box_cxcywh_to_xyxy(known_bbox_expand_pos), box_cxcywh_to_xyxy(boxes))[0]
                    pos_bbox_ = known_bbox_[positive_idx]
                    ious = box_iou(pos_bbox_, box_cxcywh_to_xyxy(boxes))[0]
                    ious[~valid_mask] = -100 # mask out the cross-image bbox
                    ious_label = ious.max(-1)[1]
                    need_check = (ious_label != pos_bbox_iou_label)
                    if need_check.sum() :
                        pos_rand_part[need_check] /= 2
                        known_bbox_[positive_idx]= pos_bbox_pre + torch.mul(pos_rand_part,
                                                            pos_diff).cuda() * box_noise_scale
                    else :
                        break
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        # known_bbox_expand[:num_pos_per_group]
        # known_labels_expaned[:num_pos_per_group]
        
        m = known_labels_expaned.long().to('cuda')
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        # filter the negative part
        if not use_cdn :
            positive_idx = torch.tensor(range(single_pad)).long().cuda().unsqueeze(0).repeat(dn_number, 1)
            positive_idx += (torch.tensor(range(dn_number)) * single_pad * 2).long().cuda().unsqueeze(1)
            positive_idx = positive_idx.flatten(0)
            input_query_label = input_query_label[:, positive_idx]
            input_query_bbox = input_query_bbox[:, positive_idx]
            pad_size = pad_size // 2
            group_pad = single_pad
        else :
            group_pad = single_pad * 2

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other

        # for i in range(dn_number):
        #     if i == 0:
        #         attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
        #     if i == dn_number - 1:
        #         attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
        #     else:
        #         attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
        #         attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        for i in range(dn_number):
            if i == 0:
                attn_mask[group_pad * i:group_pad * (i + 1), group_pad * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[group_pad * i:group_pad * (i + 1), :group_pad * i] = True
            else:
                attn_mask[group_pad * i:group_pad * (i + 1), group_pad * (i + 1):pad_size] = True
                attn_mask[group_pad * i:group_pad * (i + 1), :group_pad * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
            'use_cdn': use_cdn
        }
    else:

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta['output_known_lbs_bboxes'] = out
    return outputs_class, outputs_coord


