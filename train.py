import datetime
import time
import os
import json
import logging
import numpy as np
import random
import torch.utils.data
import torch.backends.cudnn
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile

from configs.defaults import _C as cfg
from encoding_custom.optimizers import get_optimizer, get_lr_scheduler
from encoding_custom.losses.mtl_loss import MTLLoss
from utilities import metric_utils, train_utils, dist_utils, generic_utils
from encoding_custom.evaluation.evaluators import get_mtl_evaluator


class Trainer:

    def __init__(self, args, logger):
        self.args = args
        dir_name = f'{args.checkname}_{cfg.MODEL.NAME}_{cfg.MODEL.BACKBONE_NAME}_%02d' % args.batch_size
        self.root_dir = os.path.join(args.output_dir, f'{args.dataset}/{dir_name}')
        train_utils.mkdir(self.root_dir)
        copyfile(args.config_file, os.path.join(self.root_dir, args.config_file.split('/')[-1]))

        if args.distributed:
            dist_utils.init_distributed_mode(args)
            if not dist_utils.is_main_process():
                args.pretrained = False

        self.log_dir = os.path.join(self.root_dir, "summary")
        if dist_utils.is_main_process():
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None
            logger.disabled = True

        self.tasks = [task for task, status in dict(cfg.TASKS_DICT).items() if status]

        with open(os.path.join(self.root_dir, 'config.json'), 'w') as fp:
            argparse_dict = vars(args)
            json.dump(argparse_dict, fp, sort_keys=True, indent=4)

        self.device = torch.device(args.device)
        self.norm_layer = generic_utils.get_norm_layer(cfg, args)

        dataset_train, dataset_val = train_utils.get_dataset(cfg, args)
        self.dl_train, self.dl_val = train_utils.get_dataloader(
            cfg, args, dataset_train, dataset_val)

        self.model = generic_utils.get_model(cfg, args, self.norm_layer, self.tasks)
        self.model.to(self.device)

        self.mtl_loss = MTLLoss(cfg, dict(cfg.TASKS_DICT), args.epochs, args.batch_size)
        self.mtl_loss.to(self.device)

        self.optimizer = None
        self.lr_scheduler = None
        self.init_optimizer()
        self.init_scheduler()

        self.start_epoch = train_utils.load_model(args, self.model, self.root_dir,
                                                  optimizer=self.optimizer)

        if args.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[args.gpu], output_device=args.gpu,
                find_unused_parameters=args.find_unused_params)
            if dist_utils.is_main_process():
                logging.info('Model converted to DDP model with {} '
                             'cuda devices'.format(torch.cuda.device_count()))

        self.task_to_min_or_max = dict(cfg.TASK_TO_MIN_OR_MAX)

    def init_optimizer(self):
        params_to_optimize = [
            {"params": [p for p in self.model.parameters() if p.requires_grad]},
            {"params": [p for p in self.mtl_loss.parameters() if p.requires_grad]}]

        self.optimizer = get_optimizer(self.args, params_to_optimize, cfg)

    def init_scheduler(self):
        self.lr_scheduler = get_lr_scheduler(self.args, self.optimizer, cfg)

    def eval_and_save(self, epoch, saver):
        task_to_metrics = {task: None for task in self.tasks}
        if epoch == 1 or epoch % self.args.eval_frequency == 0 or epoch == self.args.epochs:
            val_metric_logger = self.evaluate(epoch=epoch)
            for task in self.tasks:
                if val_metric_logger is not None:
                    key_metric = [key for key in val_metric_logger.meters.keys()
                                  if f'key_metrics/{task}' in key]
                    if len(key_metric) == 1:
                        task_to_metrics[task] = \
                            val_metric_logger.meters[key_metric[0]].global_avg

        save_dict = {'optimizer': self.optimizer.state_dict(),
                     'epoch': epoch, 'args': self.args}
        if not self.args.distributed:
            save_dict.update({'model': self.model.state_dict()})
        else:
            save_dict.update({'model': self.model.module.state_dict()})

        if dist_utils.is_main_process():
            saver.save_models(save_dict, epoch, task_to_metrics)

    def train_and_evaluate(self):
        saver = train_utils.ModelSaver(cfg, self.root_dir, self.tasks, self.task_to_min_or_max)
        start_time = time.time()
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            self.train_one_epoch(epoch)
            self.eval_and_save(epoch, saver)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def loss_stub(self, loss, loss_dict, predictions, targets):
        return loss, loss_dict

    def train_one_epoch(self, epoch):
        self.model.train()
        metric_logger = metric_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('misc/learning_rate', metric_utils.SmoothedValue(
            window_size=1, fmt='{value}'))
        header = f'Epoch: [{epoch}]'

        for cnt, (images, targets) in enumerate(
                metric_logger.log_every(self.dl_train, self.args.print_freq, header)):
            images, targets = train_utils.move_tensors_to_device(images, targets,
                                                                 self.device)
            predictions = self.model(images, targets=targets)
            loss, loss_dict = self.mtl_loss(predictions, targets, epoch=epoch)
            loss, loss_dict = self.loss_stub(loss, loss_dict, predictions, targets)
            metric_logger = train_utils.update_logger(loss_dict, metric_logger)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            metric_logger.update(**{
                'losses/total_loss': loss.item(),
                'misc/learning_rate': self.optimizer.param_groups[0]["lr"]})

        self.lr_scheduler.step(epoch=None)
        if self.writer is not None:
            train_utils.log_meters_in_tb(self.writer, metric_logger, epoch, 'train')

    def evaluate(self, epoch=None):
        if epoch is None:
            epoch = self.start_epoch
        self.model.eval()
        val_metric_logger = metric_utils.MetricLogger(delimiter="  ")
        header = 'Validate:'
        mtl_evaluator = get_mtl_evaluator(cfg, self.tasks, self.dl_val.dataset, self.root_dir)

        with torch.no_grad():
            for cnt, (images, targets, image_idxs) in enumerate(
                    val_metric_logger.log_every(self.dl_val, self.args.print_freq, header)):
                images, targets = train_utils.move_tensors_to_device(images, targets,
                                                                     self.device)

                predictions = self.model(images)
                loss, loss_dict = self.mtl_loss(predictions, targets)
                loss = loss.mean()
                val_metric_logger = train_utils.update_logger(loss_dict, val_metric_logger)
                val_metric_logger.update(**{'losses/total_loss': loss.item()})

                mtl_evaluator.process(targets, predictions, image_idxs)

        val_metric_logger.update(**mtl_evaluator.evaluate())
        if self.writer is not None:
            train_utils.log_meters_in_tb(self.writer, val_metric_logger, epoch, 'val')

        train_utils.log_and_dump_metrics(val_metric_logger,
                                         path=self.root_dir, epoch=epoch)
        return val_metric_logger


def main():
    # os.environ["RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "2"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6666"

    parser = train_utils.parse_args()
    args = parser.parse_args()
    train_utils.update_config_node(cfg, args)
    cfg.freeze()
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(args.seed)
        np.random.seed(args.seed)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    trainer = Trainer(args, logger)
    if args.eval_only:
        trainer.evaluate()
    else:
        trainer.train_and_evaluate()


if __name__ == "__main__":
    main()
