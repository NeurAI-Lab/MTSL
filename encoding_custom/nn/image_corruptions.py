import os
import logging
import argparse
import numpy as np
import csv
import cv2
import torch
from imagecorruptions import get_corruption_names

from configs.defaults import _C as cfg
from utilities import metric_utils, train_utils
from utilities.infer_utils import setup_infer_model
from utilities.generic_utils import back_transform
from encoding_custom.evaluation.evaluators import get_mtl_evaluator


def parse_ood_args(parser=None, add_common_args=True):
    if parser is None:
        parser = argparse.ArgumentParser(description='OOD Tests..')

    if add_common_args:
        parser = train_utils.add_common_args(parser)

    parser.add_argument(
        '--corrupted-data-path', type=str, default=None,
        help='path containing the corrupted data..')

    return parser


class Corruptions:

    def __init__(self, arguments, model, tasks, dl_val, device, get_dicts=False):
        self.model = model
        self.tasks = tasks
        self.dl_val = dl_val
        self.device = device
        self.get_dicts = get_dicts
        self.args = arguments
        self.print_freq = arguments.print_freq
        self.root_dir = None
        if not get_dicts:
            dir_name = f'{arguments.checkname}_{cfg.MODEL.NAME}_'
            dir_name += f'{cfg.MODEL.BACKBONE_NAME}_%02d' % arguments.batch_size
            self.root_dir = os.path.join(
                arguments.output_dir, f'{arguments.dataset}/{dir_name}')
            train_utils.mkdir(self.root_dir)
        self.task_to_metric = {'segment': 'key_metrics/segment_MIoU',
                               'depth': 'key_metrics/depth_rmse',
                               'sur_nor': 'key_metrics/sur_nor_cosine_similarity',
                               'sem_cont': 'key_metrics/sem_cont_bce_err',
                               'ae': 'key_metrics/ae_mse'}

    def check_images(self):
        val_metric_logger = metric_utils.MetricLogger(delimiter="  ")
        header = 'Validating Corruption:'
        for cnt, (images, targets, image_idxs) in enumerate(
                val_metric_logger.log_every(
                    self.dl_val, self.print_freq, header)):
            images = back_transform(images, cfg, scale=255.)
            images = images.permute(0, 2, 3, 1).cpu().numpy()
            image = np.array(images[0], dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('image', image)
            cv2.waitKey(0)

    def eval_model(self):
        mtl_evaluator = get_mtl_evaluator(
            cfg, self.tasks, self.dl_val.dataset, self.root_dir)
        val_metric_logger = metric_utils.MetricLogger(delimiter="  ")
        header = 'Validating Corruption:'
        with torch.no_grad():
            for cnt, (images, targets, image_idxs) in enumerate(
                    val_metric_logger.log_every(
                        self.dl_val, self.print_freq, header)):
                images, targets = train_utils.move_tensors_to_device(
                    images, targets, self.device)
                predictions = self.model(images)
                mtl_evaluator.process(targets, predictions, image_idxs)

        val_metric_logger.update(**mtl_evaluator.evaluate())
        return train_utils.log_and_dump_metrics(
            val_metric_logger, path=self.root_dir, dump=False)

    def eval_under_corruptions(self):
        cfg.TRANSFORMS_KWARGS.is_classify = True

        corruption_names = []
        odd_accuracies = {task: [] for task in self.tasks}
        for corruption in get_corruption_names():
            cfg.TRANSFORMS_KWARGS.corruption_name = corruption
            ood_acc = {task: [] for task in self.tasks}
            for severity in range(1, 6):
                cfg.TRANSFORMS_KWARGS.severity = severity
                corruption_names.append(corruption)
                # This transform edits the transform
                # attribute in the torchvision dataloader..
                logging.info(f'Current run: Corruption {corruption}, '
                             f'Severity {severity}')
                self.update_dataset_paths(corruption, severity)
                results = self.eval_model()
                for task in self.tasks:
                    ood_acc[task].append(
                        round(results[self.task_to_metric[task]], 4))
            for task in self.tasks:
                odd_accuracies[task].append(ood_acc[task])

        if self.get_dicts:
            return corruption_names, odd_accuracies
        with open(os.path.join(self.root_dir, 'ood_result.csv'), 'w',
                  newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(corruption_names)
            for task in self.tasks:
                writer.writerow(sum(odd_accuracies[task], []))

    def update_dataset_paths(self, corruption, severity):
        self.dl_val.dataset.image_paths = {file_name: os.path.join(
            self.args.corrupted_data_path, corruption, str(severity), file_name)
            for file_name in self.dl_val.dataset.image_paths.keys()}


def main():
    logging.getLogger().setLevel(logging.INFO)

    model, tasks, dl_val, device = setup_infer_model(
        args, cfg, num_workers=args.workers, batch_size=args.batch_size,
        collate_fn='train_collate', freeze_cfg=False,
        data_kwargs={'mode': 'val'})

    corruptor = Corruptions(args, model, tasks, dl_val, device)
    corruptor.eval_under_corruptions()


if __name__ == "__main__":
    args = parse_ood_args().parse_args()
    args.split = 'val'
    main()
