import errno
import os
import uuid
import json
import logging
import argparse
import yaml
from natsort import natsorted
import glob
import copy
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from detectron2.structures import ImageList

from configs import dict_configs
from encoding_custom.datasets import get_num_classes, get_instance_names,\
    get_semantic_names
from encoding_custom.datasets import get_multitask_dataset
from utilities import dist_utils


def add_common_args(parser):
    parser.add_argument(
        "--config-file", default="", metavar="FILE",
        help="path to config file", type=str)
    parser.add_argument(
        '--seed', default=0, type=int,
        help='Random seed for processes for fixing the seed. '
             'if you want to disable it Enter negative number')

    # dataset...
    parser.add_argument(
        '--dataset', default='uninet_cs', help='dataset')
    parser.add_argument(
        '--data-folder', type=str,
        default=os.path.join('/data/input/datasets/Cityscapes'),
        help='training dataset folder')

    # training params...
    parser.add_argument(
        '--device', default='cuda', help='device')
    parser.add_argument(
        '-j', '--workers', default=8, type=int, metavar='N',
        help='number of data loading workers (default: 8)')
    parser.add_argument(
        '-b', '--batch-size', default=1, type=int)
    parser.add_argument(
        '--crop-size', type=int, default=[512, 1024], nargs="*",
        help='crop image size')

    # misc
    parser.add_argument(
        '--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument(
        '--output-dir', default='./runs', help='path where to save')
    parser.add_argument(
        '--checkname', type=str, default='init_model_' + str(uuid.uuid4()),
        help='set the checkpoint name')
    parser.add_argument(
        '--resume', default=None, help='resume from checkpoint')
    parser.add_argument(
        "--log-per-class-metrics", default=False,
        help="log per class metrics in tensorboard", action="store_true")

    return parser


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Multitask Training')
    parser = add_common_args(parser)
    parser.add_argument(
        "--eval-only", dest="eval_only", help="Only test the model",
        action="store_true")

    # training params...
    parser.add_argument(
        '--epochs', default=140, type=int, metavar='N',
        help='number of total epochs to run')
    parser.add_argument(
        "--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo",
        action="store_true")

    # Learning rate...
    parser.add_argument(
        '--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument(
        '--lr-strategy', default="stepwise", type=str, help='learning rate strategy')
    parser.add_argument(
        '--lr-decay', default=[98, 126], nargs="*", type=int,
        help='steps for learning rate decay if it is stepwise')

    # Optimizer...
    parser.add_argument(
        '--base-optimizer', type=str, default="Adam", help='optimizer')
    parser.add_argument(
        '--wrap-optimizer', type=str, default=None, help='optimizer')
    parser.add_argument(
        '--wd', '--weight-decay', default=5e-05, type=float, metavar='W',
        help='weight decay (default: 5e-05)', dest='weight_decay')

    # distributed training parameters..
    parser.add_argument(
        '--distributed', action='store_true', help='go multi gpu', default=False)
    parser.add_argument(
        '--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument(
        '--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument(
        '--local_rank', default=0, type=int, help='process rank on node')
    parser.add_argument(
        '--find-unused-params', action='store_true',
        help='find unused parameters in model', default=False)

    # misc...
    parser.add_argument(
        '--eval-frequency', default=2, type=int, metavar='N',
        help='epoch intervals after which eval is run')
    parser.add_argument(
        '--resume-after-suspend', default=True,
        help='skip validation during training')
    parser.add_argument(
        "--val-img-display", default=False,
        help="use tensorboard image for segmentation and depth at validation",
        action="store_true")
    parser.add_argument(
        "--train-img-display", default=False,
        help="use tensorboard image for segmentation and depth at train",
        action="store_true")

    return parser


def update_config_node(cfg, args, dict_cfg=None):
    # instance tasks also include background in total number of classes...
    num_classes = get_num_classes(args.dataset)
    num_classes = {key.upper(): val for key, val in num_classes.items()}
    cfg_dict = {name: value for name, value in dict_configs.__dict__.items()
                if not (name.startswith('__') or name.startswith('CN') or
                        name.startswith('_C'))}
    init_dict = {'NUM_CLASSES': num_classes,
                 'INSTANCE_NAMES': get_instance_names(args.dataset),
                 'SEMANTIC_NAMES': get_semantic_names(args.dataset)}
    if args.config_file != "":
        with open(args.config_file, "r") as f:
            custom_config = yaml.load(f, Loader=yaml.FullLoader)
        custom_cfg_dict = custom_config.get('DICT_CONFIG', {})
        custom_config.pop('DICT_CONFIG', None)
    else:
        custom_config = {}
        custom_cfg_dict = {}
    for key in custom_cfg_dict.keys():
        if key in cfg_dict.keys():
            cfg_dict[key].update(custom_cfg_dict[key])
        else:
            cfg_dict[key] = custom_cfg_dict[key]
    if type(dict_cfg) is dict:
        cfg_dict.update(dict_cfg)
    init_dict.update(cfg_dict)
    cfg.__init__(init_dict=init_dict)
    custom_config = cfg.load_cfg(yaml.dump(custom_config))
    cfg.merge_from_other_cfg(custom_config)
    cfg.INPUT.IMAGE_SIZE = args.crop_size
    cfg.DEVICE = args.device
    if args.device == 'cuda':
        cfg.DEVICE = f'{args.device}:{args.local_rank}'
    cfg.MISC.LOG_PER_CLASS_METRICS = args.log_per_class_metrics

    return cfg


class BatchCollator(object):
    """
    From a list of samples from the dataset, returns the batched images
     and targets. This should be passed to the DataLoader.
    """

    def __init__(self, cfg, dataset=None, val=False):
        self.cfg = cfg
        self.val = val
        self.min_elements = 1 if not val else 2
        self.dataset = dataset
        self.size_divisibility = cfg.DATALOADER.SIZE_DIVISIBILITY
        if cfg.INPUT.IMAGE_SIZE[0] <= 224:
            # avoid enforcing divisibility for cls datasets...
            self.size_divisibility = 0

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # imagelist is used to support multi scale inputs..
        # +1 is used as stride 2 is skipped..
        images = ImageList.from_tensors(
            transposed_batch[0], size_divisibility=self.size_divisibility)
        images = images.tensor

        targets = {}
        tars = transposed_batch[1]
        for task in tars[0].keys():
            ts = [d[task] for d in tars]
            if task == 'intrinsics' or task == 'classify':
                targets.update({task: default_collate(ts)})
            else:
                ts = ImageList.from_tensors(
                    ts, pad_value=-1, size_divisibility=self.size_divisibility)
                targets.update({task: ts.tensor})

        if self.val:
            idxs = default_collate(transposed_batch[2])
            return images, targets, idxs

        return images, targets


def get_dataset(cfg, args):
    data_kwargs = {'crop_size': args.crop_size, 'root': args.data_folder,
                   'cfg': cfg}

    dataset_train = get_multitask_dataset(
        args.dataset, split='train', mode='train', **data_kwargs)
    dataset_val = get_multitask_dataset(
        args.dataset, split='minival', mode='val', **data_kwargs)

    return dataset_train, dataset_val


def get_dataloader(cfg, args, dataset_train, dataset_val):
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        val_sampler = torch.utils.data.sampler.SequentialSampler(dataset_val)
    dl_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, pin_memory=True,
        collate_fn=BatchCollator(cfg, dataset=dataset_train),
        sampler=train_sampler, num_workers=args.workers, drop_last=True)

    dl_val = torch.utils.data.DataLoader(
        dataset_val, sampler=val_sampler, num_workers=args.workers,
        collate_fn=BatchCollator(cfg, dataset=dataset_val, val=True),
        pin_memory=True, batch_size=args.batch_size, drop_last=False)

    return dl_train, dl_val


def load_model(args, model, root_dir, optimizer=None):
    start_epoch = 1

    if args.resume is None and args.resume_after_suspend:
        if os.path.exists(root_dir):
            files = glob.glob(os.path.join(root_dir, '*_latest_*.pth'))
            if len(files) > 0:
                files = natsorted(files)
                args.resume = files[-1]
                logging.info(args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError:
                logging.info('Unable to load optimizer.')
        if args.resume_after_suspend and 'epoch' in checkpoint.keys():
            start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Model loaded successfully from {args.resume}..")
    else:
        logging.info('No model found in root dir..')

    return start_epoch


def move_tensors_to_device(images, targets, device):
    for task, ts in targets.items():
        if type(ts) is list:
            targets[task] = [target.to(device) for target in ts]
        else:
            targets[task] = ts.to(device)

    return images.to(device), targets


def update_logger(loss_dict, metric_logger):
    for loss_name, loss_ts in loss_dict.items():
        loss_ts = loss_ts.mean()
        metric_logger.update(**{loss_name: loss_ts.item()})

    return metric_logger


def log_meters_in_tb(writer, metric_logger, epoch, phase):
    for meter_name, value in metric_logger.meters.items():
        check_names = ['key_metrics/', 'metrics/', 'losses/',
                       'misc/', 'classwise/']
        if any([name in meter_name for name in check_names]):
            tb_name = f'{phase}_{meter_name}'
        else:
            continue
        writer.add_scalar(f'{tb_name}', value.global_avg, epoch)


def log_and_dump_metrics(val_metric_logger, path="./", epoch=-1,
                         dump_name='results', dump=True):
    results = {}
    for meter_name, value in val_metric_logger.meters.items():
        if 'key_metrics/' in meter_name or 'metrics/' in meter_name:
            results.update({meter_name: round(value.global_avg, 4)})

    results.update({'epoch': epoch})
    if dump:
        with open(os.path.join(path, f'{dump_name}.json'), 'w') as fp:
            json.dump(results, fp, sort_keys=True, indent=4)

    result_keys = list(results.keys())
    for key in result_keys:
        if 'key_metrics/' not in key:
            results.pop(key)
    logging.info(f'Validation metrics: {results}')

    return results


class ModelSaver:

    def __init__(self, cfg, save_root, tasks, task_to_min_or_max, base_name='model',
                 best_name='best', latest_name='latest'):
        self.save_root = save_root
        self.task_to_min_or_max = {task: task_to_min_or_max.get(
            task, 1) for task in tasks}
        self.base_name = base_name
        self.best_name = best_name
        self.latest_name = latest_name
        self.models_to_keep = cfg.MISC.MODELS_TO_KEEP
        self.task_to_best = {task: None for task in tasks}
        self.save_task_best = cfg.MISC.SAVE_TASK_BEST

    def read_models(self):
        # model file name: base_{task_task_.._task}_epoch.pth
        task_to_models = {task: [] for task in self.task_to_min_or_max.keys()}
        task_to_models.update({self.latest_name: []})

        for file in os.listdir(self.save_root):
            if os.path.isdir(file):
                continue
            extension = file.split('.')[-1]
            if extension != 'pth':
                continue
            names = file.split('_')[1:-1]
            task = '_'.join(names)
            task_to_models[task].append(file)

        return task_to_models

    def delete_old(self, task_to_models):
        for models in task_to_models.values():
            models = sorted(models)
            if len(models) > self.models_to_keep:
                for m in models[:-self.models_to_keep]:
                    os.remove(os.path.join(self.save_root, m))

    def is_best(self, task_to_metrics):
        task_to_is_best = {task: False for task in task_to_metrics.keys()}

        for task, metric in task_to_metrics.items():
            if metric is None:
                continue
            current_best = self.task_to_best[task]
            min_or_max = self.task_to_min_or_max[task]
            if current_best is None or (min_or_max == 1 and metric > current_best) or (
                    min_or_max == -1 and metric < current_best):
                task_to_is_best[task] = True
                self.task_to_best[task] = metric

        return task_to_is_best

    def save(self, save_dict, epoch, task):
        file_name = f'{self.base_name}_{task}_%03d.pth' % epoch
        dist_utils.save_on_master(save_dict, os.path.join(self.save_root, file_name),
                                  _use_new_zipfile_serialization=False)

    def save_models(self, save_dict, epoch, task_to_metrics):
        if self.save_task_best:
            task_to_is_best = self.is_best(task_to_metrics)
            for task, is_best in task_to_is_best.items():
                if is_best:
                    self.save(save_dict, epoch, task)

        self.save(save_dict, epoch, self.latest_name)
        task_to_models = self.read_models()
        self.delete_old(task_to_models)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
