from torch import nn
import torch

from encoding_custom.losses import task_to_loss_fn


class MTLLoss(nn.Module):

    def __init__(self, cfg, tasks_dict, n_epochs=1, batch_size=1):
        super(MTLLoss, self).__init__()
        self.batch_size = batch_size
        self.cfg = cfg

        task_to_loss_name = dict(cfg.TASK_TO_LOSS_NAME)
        task_to_loss_args = dict(cfg.TASK_TO_LOSS_ARGS)
        task_to_loss_kwargs = dict(cfg.TASK_TO_LOSS_KWARGS)
        self.task_to_call_kwargs = dict(cfg.TASK_TO_LOSS_CALL_KWARGS)

        self.tasks = [task for task, status in tasks_dict.items() if status]
        # all losses have access to what tasks are predicted..
        for task in tasks_dict.keys():
            if task in task_to_loss_kwargs.keys():
                task_to_loss_kwargs[task].update(tasks_dict)
            task_to_loss_kwargs.update({task: tasks_dict})

        self.task_to_fn = {}
        for task, status in tasks_dict.items():
            if not status:
                continue
            loss_name = task_to_loss_name.get(task, 'default')
            loss_fn = task_to_loss_fn[task][loss_name]
            if loss_fn is None:
                continue
            self.task_to_fn.update({task: loss_fn(
                cfg, *task_to_loss_args.get(task, []),
                **task_to_loss_kwargs.get(task, {}))})

        self.loss_to_weights = dict(cfg.LOSS_INIT_WEIGHTS)
        self.loss_to_weights = {'losses/' + key: value for key, value
                                in self.loss_to_weights.items()}
        self.loss_to_start = dict(cfg.LOSS_START_EPOCH)
        self.loss_to_start = {'losses/' + key: value for key, value
                              in self.loss_to_start.items()}

    @staticmethod
    def get_losses(loss_dict):
        losses = []
        loss_names = []
        for loss_name, curr_loss in loss_dict.items():
            if 'losses' not in loss_name:
                continue
            losses.append(curr_loss)
            loss_names.append(loss_name)
        return torch.stack(losses), loss_names

    def epoch_start(self, loss_name, curr_loss, epoch):
        start_epoch = self.loss_to_start.get(loss_name, 0)
        ep_w = 1 if epoch >= start_epoch else 0
        curr_loss = curr_loss * ep_w

        return curr_loss

    def handcrafted_balancing(self, losses_to_add):
        loss = 0
        for loss_name, curr_loss in losses_to_add.items():
            curr_loss = curr_loss * self.loss_to_weights.get(loss_name, 1)
            loss = loss + curr_loss
            losses_to_add[loss_name] = curr_loss

        return loss, losses_to_add

    def loss_balancing(self, losses_to_add, epoch):
        loss, losses_to_add = self.handcrafted_balancing(losses_to_add)
        return loss, losses_to_add

    def check_losses(self, task, task_losses, loss_dict, losses_to_add, epoch):
        if type(task_losses) is not dict:
            task_losses = {task: {f'losses/{task}_loss': task_losses}}

        for losses in task_losses.values():
            for loss_name, loss_ts in losses.items():
                if 'losses/' in loss_name:
                    curr_loss = self.epoch_start(loss_name, loss_ts, epoch)
                    losses[loss_name] = curr_loss
                    losses_to_add.update({loss_name: curr_loss})
                else:
                    loss_dict.update({loss_name: loss_ts})

        return loss_dict, losses_to_add

    def forward_mtsl(self, predictions, targets, epoch=1):
        loss_dict = {}
        losses_to_add = {}
        for task, fn in self.task_to_fn.items():
            task_losses = fn(predictions[task], targets.get(task, targets),
                             **self.task_to_call_kwargs.get(task, {}))
            if task_losses is None:
                continue
            loss_dict, losses_to_add = self.check_losses(
                task, task_losses, loss_dict, losses_to_add, epoch)

        return loss_dict, losses_to_add

    def forward(self, predictions, targets, epoch=1):
        loss_dict, losses_to_add = self.forward_mtsl(
            predictions, targets, epoch)
        loss, losses_to_add = self.loss_balancing(losses_to_add, epoch)
        loss_dict.update(losses_to_add)
        return loss, loss_dict
