# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json



import random
import time
from pathlib import Path
import os, sys
from typing import Optional
from util.get_param_dicts import get_param_dict




from util.logger import setup_logger

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

import dataset
import util.misc as utils
from dataset import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test
import models
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
from dataset.samplers import RepeatFactorTrainingSampler, ClassAwareSampler

class MultiDatasetLoader():
    def __init__(self, main_loader, sub_loader, main_weight=1, sub_weight=1):
        self.main_loader = main_loader
        self.sub_loader = sub_loader
        self.main_weight = main_weight
        self.sub_weight = sub_weight
        self.total_weight = main_weight + sub_weight

    def __len__(self):
        num_main_loader = len(self.main_loader)
        return num_main_loader + num_main_loader//self.main_weight*self.sub_weight + num_main_loader%self.main_weight
    
    def set_epoch(self, epoch):
        def _set_epoch(loader, epoch):
            if hasattr(loader, 'set_epoch'):
                loader.set_epoch(epoch)
        _set_epoch(self.main_loader, epoch)
        _set_epoch(self.sub_loader, epoch)

    def __iter__(self):
        count = 0
        main_iter = iter(self.main_loader)
        sub_iter = iter(self.sub_loader)
        while True :
            if count <self.main_weight :
                # support python>=3.7
                try :
                    yield next(main_iter)
                except StopIteration:
                    return
            else :
                try :
                    yield next(sub_iter)
                except :
                    sub_iter = iter(self.sub_loader)
                    yield next(sub_iter)
            count +=1
            count %= self.total_weight


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='lvis')
    parser.add_argument('--data_path', type=str, default='DATASET/lvis')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')


    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--finetune_ignore_all', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def main(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.local_rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')


    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    try :
        use_rfs = args.use_rfs
    except :
        use_rfs = False
    if use_rfs :
        repeat_threshold = args.rfs_repeat_sh
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(dataset_train.data, repeat_threshold)
        sampler_train = RepeatFactorTrainingSampler(repeat_factors, len_dataset=len(dataset_train))

    if args.use_cas :
        def convert_coco_to_datadict(dataset):
            coco = dataset.coco
            dataset_list = []
            for i in range(len(dataset)):
                img_id = dataset.ids[i]
                if isinstance(img_id, str):
                    img_id = [img_id]
                ann_ids = coco.getAnnIds(imgIds=img_id)
                target = coco.loadAnns(ann_ids)
                dataset_list.append({'annotations': target})
            return dataset_list

        dataset_dict = convert_coco_to_datadict(dataset_train)
        sampler_train = ClassAwareSampler(dataset_dict)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False,
                                 num_workers=args.num_workers,
                                 collate_fn=utils.collate_fn_tta if args.use_tta else utils.collate_fn, 
                                 )

    try :
        use_imagenet = args.use_imagenet
    except :
        use_imagenet = False
    try :
        use_imagenet_v3det = args.use_imagenet_v3det
    except :
        use_imagenet_v3det = False
    try :
        use_extra_data = args.use_extra_data
    except :
        use_extra_data = False

    if use_imagenet or use_extra_data or use_imagenet_v3det :
        if use_imagenet :
            data_extra = build_dataset(image_set='train', args=args, imagenet_lvis=True)
        elif use_imagenet_v3det:
            data_extra = build_dataset(image_set='train', args=args, imagenet_v3det=True)
        else :
            extra_data_type = args.extra_data_type
            from dataset.coco import build_extra_cocostyle_data
            if extra_data_type == 'unlabeled_inet21k':
                data_extra = build_extra_cocostyle_data('DATASET/imagenet-21k/', 'unzip', 'images.json',
                                                            False, args, is_image_folder=True)
            elif extra_data_type == 'imagenet21_od':
                data_extra = build_extra_cocostyle_data('DATASET/imagenet21-od', 'images', 'im22k_train_t0.6.json',
                                                            False, args, is_image_aug=True)
            elif extra_data_type == 'o365':
                data_extra = build_extra_cocostyle_data('DATASET/object365', 'train', 'objects365_train.json',
                                                            False, args)
            elif extra_data_type == 'cc3m':
                data_extra = build_extra_cocostyle_data('DATASET/gcc', 'CC3M', 'CC3M/cc3m_zip_json.json',
                                                            False, args, is_image_folder=True)
            else :
                raise NotImplementedError

        if args.distributed :
            sampler_extra = DistributedSampler(data_extra)
        else :
            sampler_extra = torch.utils.data.RandomSampler(data_extra)
        batch_sampler_extra = torch.utils.data.BatchSampler(
            sampler_extra, args.batch_size, drop_last=True)
        data_loader_extra = DataLoader(data_extra, batch_sampler=batch_sampler_extra,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_train = MultiDatasetLoader(data_loader_train, data_loader_extra, main_weight=args.main_weight, sub_weight=args.sub_weight)

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = dataset.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    elif 'v3det' in args.dataset_file:
        coco_val = dataset.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        # from collections import OrderedDict
        # new_model = OrderedDict()
        # model_state_dict = model_without_ddp.state_dict()
        # for k, v in checkpoint['model'].items() :
        #     if model_state_dict[k].shape == v.shape:
        #         new_model[k] = v
        #     else :
        #         print(k)
        # model_without_ddp.load_state_dict(new_model, strict=False)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)                

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

            lr_scheduler.step_size = args.lr_drop

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        if args.finetune_ignore_all :
            _model_state_dict = model_without_ddp.state_dict()
            _tmp_st = OrderedDict()
            for k, v in utils.clean_state_dict(checkpoint).items() :
                if k in _model_state_dict and _model_state_dict[k].shape == v.shape:
                    _tmp_st[k] = v
        else :
            _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)        

    if args.test :
        model_without_ddp.set_cats(dataset_val.cats)
        result = test(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        from detectron2.utils.comm import gather
        import itertools
        result = gather(result)
        result = list(itertools.chain(*result))
        if args.output_dir and utils.is_main_process():
            with (output_dir / "bbox_pred.json").open("w") as f:
                json.dump(result, f)
        return

    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        model_without_ddp.set_cats(dataset_val.cats)
        criterion.set_cats(dataset_val.cats)
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        model_without_ddp.set_cats(dataset_train.cats)
        criterion.set_cats(dataset_train.cats)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                print('save ckpt: ', checkpoint_path)
                utils.save_on_master(weights, checkpoint_path)
                
        # eval
        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.epochs :
            model_without_ddp.set_cats(dataset_val.cats)
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            map_regular = test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        else :
            test_stats = {}
            coco_evaluator = None
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }

        # eval ema
        if args.use_ema:
            ema_test_stats, ema_coco_evaluator = evaluate(
                ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
            map_ema = ema_test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                utils.save_on_master({
                    'model': ema_m.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        log_stats.update(best_map_holder.summary())

        ep_paras = {
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from dataset.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
