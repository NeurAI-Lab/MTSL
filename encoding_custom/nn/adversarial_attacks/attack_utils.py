import os
import cv2
import numpy as np
import torch
from abc import abstractmethod
import logging
import argparse

from encoding_custom.losses.mtl_loss import MTLLoss
from utilities.generic_utils import back_transform
from utilities import viz_utils, generic_utils, train_utils

colors = np.array(viz_utils.label_colours_global, dtype=np.uint8)


class BaseAttacker:
    added_additional_args = False

    def __init__(self, args, cfg, model, tasks, dl_val, device, **kwargs):
        self.model = model
        self.tasks = tasks
        self.dl_val = dl_val
        self.device = device
        self.kwargs = kwargs
        self.cfg = cfg

        self.eps = args.eps
        self.alpha = args.alpha
        num_iters = kwargs.get('num_iters', None)
        if num_iters is None:
            num_iters = np.min([args.eps + 4, 1.25 * args.eps])
            num_iters = int(np.max([np.ceil(num_iters), 1]))
        self.num_iters = num_iters
        logging.info(f'Number of attack iterations: {self.num_iters}')
        self.image_mean = cfg.DATALOADER.MEAN
        self.image_std = cfg.DATALOADER.STD
        std_ts = np.array(self.image_std)
        std_ts = torch.from_numpy(std_ts).float().to(cfg.DEVICE)
        self.std_ts = std_ts[None, :, None, None]
        step_size = self.alpha / 255.
        self.image_shape = [args.batch_size, 3] + cfg.INPUT.IMAGE_SIZE
        ones_x = torch.ones(*self.image_shape).to(cfg.DEVICE)
        self.step_size_ts = ones_x * step_size / self.std_ts

        upper_bound = torch.ones(*self.image_shape).to(cfg.DEVICE)
        lower_bound = torch.zeros(*self.image_shape).to(cfg.DEVICE)
        self.upper_bound = generic_utils.forward_transform(upper_bound, cfg)
        self.lower_bound = generic_utils.forward_transform(lower_bound, cfg)

        self.mtl_loss = MTLLoss(cfg, dict(cfg.TASKS_DICT), batch_size=1)
        self.use_city_colors = args.dataset == 'uninet_cs'

    def last_batch_drop(self, images):
        last_batch_size = images.shape[0]
        self.step_size_ts = self.step_size_ts[0:last_batch_size]
        self.upper_bound = self.upper_bound[0:last_batch_size]
        self.lower_bound = self.lower_bound[0:last_batch_size]

    def check_attack_validity(self, images, targets):
        return True

    def preprocess_images(self, images, *arguments):
        return images

    def preprocess_targets(self, images, targets):
        return targets

    @abstractmethod
    def attack_objective(self, predictions, targets):
        pass

    def perform_attack(self, images, targets, **kwargs):
        adv_images = images.clone()
        upper_bound, lower_bound = get_eps_bounds(
            self.upper_bound, self.lower_bound, self.eps, self.std_ts, images)
        if not self.check_attack_validity(images, targets):
            return adv_images.detach()
        adv_images = self.preprocess_images(
            adv_images, lower_bound, upper_bound)
        targets = self.preprocess_targets(images, targets)

        for i in range(self.num_iters):
            adv_images.requires_grad = True
            predictions = self.model(adv_images)
            loss = self.attack_objective(predictions, targets)
            adv_images = self.perturb_image(
                adv_images, loss, upper_bound, lower_bound)

        return adv_images.detach()

    def perturb_image(self, adv_images, loss, upper_bound, lower_bound,
                      patch=None):
        self.model.zero_grad()
        if adv_images.grad is not None:
            adv_images.grad.data.fill_(0)
        adv_images.retain_grad()
        loss.mean().backward()

        perturb = self.step_size_ts * torch.sign(adv_images.grad)
        if patch is not None:
            perturb = perturb * patch
        adv_images = adv_images + perturb
        adv_images = clamp_tensor(
            adv_images, upper_bound, lower_bound)
        return adv_images.detach()

    @classmethod
    def additional_args(cls, parser):
        if BaseAttacker.added_additional_args:
            return parser
        # attack params...
        parser.add_argument(
            '--eps', default=0.25, type=float, help='epsilon')
        parser.add_argument(
            '--alpha', default=1., type=float, help='alpha')

        parser.add_argument(
            '--dont-eval', default=False, help='dont evaluate model with attack',
            action="store_true")

        # visualization...
        parser.add_argument(
            "--visualise", dest="visualise", help="visualise results",
            action="store_true")
        parser.add_argument(
            '--viz-save-path', type=str,
            default=None, help='path to save visualizations')
        parser.add_argument(
            '--save-adv-image', dest="save_adv_image", action="store_true",
            help='whether to save adversarial images alone; '
                 'by default a viz with predictions will be saved')

        BaseAttacker.added_additional_args = True
        return parser


def parse_args(known_attackers, parser=None, add_common_args=True):
    if parser is None:
        parser = argparse.ArgumentParser(
            description='PyTorch Multitask Adversarial Attacks')

    if add_common_args:
        parser = train_utils.add_common_args(parser)
    parser.add_argument(
        '--attack-name', default='pgd', type=str, help='attack name')

    for attacker in known_attackers:
        parser = attacker.additional_args(parser)

    return parser


def clamp_tensor(image, upper_bound, lower_bound):
    image = torch.where(image > upper_bound, upper_bound, image)
    image = torch.where(image < lower_bound, lower_bound, image)
    return image


def get_eps_bounds(upper_bound, lower_bound, eps, std_ts, images):
    epsilon = eps / 255.
    pert_epsilon = torch.ones_like(images) * epsilon / std_ts
    pert_upper = images + pert_epsilon
    pert_lower = images - pert_epsilon
    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    return upper_bound, lower_bound


def one_hot(targets, num_classes):
    targets_extend = targets.clone()
    targets_extend[targets_extend == -1] = num_classes
    targets_extend.unsqueeze_(1)  # convert to Nx1xHxW
    if targets.ndim == 3:
        oh = torch.zeros(targets.shape[0], num_classes + 1, targets.shape[1],
                         targets.shape[2]).to(targets.device)
    elif targets.ndim == 1:
        oh = torch.zeros(targets.shape[0], num_classes + 1).to(targets.device)
    else:
        raise ValueError('Unknown target dimension...')
    oh.scatter_(1, targets_extend.data, 1)
    return oh[:, :num_classes]
