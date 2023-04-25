import os
import logging
import torch.utils.data
import torch.backends.cudnn

from encoding_custom.nn.adversarial_attacks import *
from encoding_custom.nn.adversarial_attacks import attack_utils
from configs.defaults import _C as cfg
from utilities import metric_utils, train_utils
from utilities.infer_utils import setup_infer_model
from encoding_custom.evaluation.evaluators import get_mtl_evaluator


name_to_attacker = {'pgd': PGDAttack}


class AttackWrapper:
    def __init__(self, args, model, tasks, dl_val, device, attacker):
        self.args = args
        attack_args = getattr(cfg, 'ATTACK_ARGS', {})
        self.attacker = attacker(args, cfg, model, tasks, dl_val, device,
                                 **attack_args)
        self.model = model
        self.tasks = tasks
        self.dl_val = dl_val
        self.device = device
        self.print_freq = args.print_freq
        dir_name = f'{args.checkname}_{cfg.MODEL.BACKBONE_NAME}'
        self.root_dir = os.path.join(
            args.output_dir, f'{args.dataset}/{cfg.MODEL.NAME}/{dir_name}')
        train_utils.mkdir(self.root_dir)
        self.val_metric_logger = metric_utils.MetricLogger(delimiter="  ")
        self.mtl_evaluator = get_mtl_evaluator(
            cfg, self.tasks, self.dl_val.dataset, self.root_dir)

    def run(self):
        for cnt, (images, targets, image_idxs) in enumerate(
                self.val_metric_logger.log_every(
                    self.dl_val, self.print_freq, 'Running Attack:')):
            if images.shape[0] != self.args.batch_size:
                self.attacker.last_batch_drop(images)
            images, targets = train_utils.move_tensors_to_device(
                images, targets, self.device)
            adv_images = self.attacker.perform_attack(images, targets)

            if not self.args.dont_eval:
                self.eval(adv_images, targets, image_idxs)

        if not self.args.dont_eval:
            self.val_metric_logger.update(**self.mtl_evaluator.evaluate())
            return train_utils.log_and_dump_metrics(
                self.val_metric_logger, path=self.root_dir,
                dump_name='adv_performance')

    def infer(self, images):
        with torch.no_grad():
            preds = self.model(images)
        return preds

    def eval(self, adv_images, targets, image_idxs):
        with torch.no_grad():
            adv_preds = self.infer(adv_images)
            _, loss_dict = self.attacker.mtl_loss(adv_preds, targets)
            self.val_metric_logger = train_utils.update_logger(
                loss_dict, self.val_metric_logger)
            self.mtl_evaluator.process(targets, adv_preds, image_idxs)

        return adv_preds


def main():
    parser = attack_utils.parse_args(name_to_attacker.values())
    args = parser.parse_args()
    args.split = 'val'
    attack_name = args.attack_name
    attacker = name_to_attacker.get(attack_name, None)
    if attacker is None:
        raise ValueError('Unknown attacker...')

    logging.getLogger().setLevel(logging.INFO)
    model, tasks, dl_val, device = setup_infer_model(
        args, cfg, num_workers=args.workers, batch_size=args.batch_size,
        collate_fn='train_collate', data_kwargs={'mode': 'val'})
    attack_wrap = AttackWrapper(args, model, tasks, dl_val, device, attacker)
    attack_wrap.run()


if __name__ == "__main__":
    main()
