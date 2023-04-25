from collections import OrderedDict
from functools import reduce
import numpy as np
from yacs.config import CfgNode

from torch import nn
from torch.nn import SyncBatchNorm

from encoding_custom.models import get_multitask_model


def get_module(model, module):
    return reduce(getattr, module.split("."), model)


def register_forward_hooks(model, layer_names):
    def get_activations(name):
        def forward_hook(module, inp, output=None):
            acts_out[name] = output
        return forward_hook

    acts_out = OrderedDict()
    hooks = []
    layer_modules = [get_module(model, layer) for layer in layer_names]
    for layer, layer_name in zip(layer_modules, layer_names):
        hooks.append(layer.register_forward_hook(
            get_activations(layer_name)))

    return hooks, acts_out


def get_norm_layer(cfg, args):
    if cfg.MODEL.NORM_LAYER == 'bn':
        norm_layer = SyncBatchNorm if args.distributed else nn.BatchNorm2d
    else:
        raise ValueError("norm layer not found")

    return norm_layer


def get_model(cfg, args, norm_layer, tasks, **kwargs):
    model = get_multitask_model(
        cfg.MODEL.NAME, backbone=cfg.MODEL.BACKBONE_NAME, tasks=tasks,
        norm_layer=norm_layer, cfg=cfg, pretrained=args.pretrained, **kwargs)

    return model


def back_transform(image, cfg, scale=1.):
    image[:, 0, :, :] = (image[:, 0, :, :] *
                         cfg.DATALOADER.STD[0]) + cfg.DATALOADER.MEAN[0]
    image[:, 1, :, :] = (image[:, 1, :, :] *
                         cfg.DATALOADER.STD[1]) + cfg.DATALOADER.MEAN[1]
    image[:, 2, :, :] = (image[:, 2, :, :] *
                         cfg.DATALOADER.STD[2]) + cfg.DATALOADER.MEAN[2]
    return image * scale


def forward_transform(image, cfg, scale=1.):
    image = image / scale
    image[:, 0, :, :] = (image[:, 0, :, :] -
                         cfg.DATALOADER.MEAN[0]) / cfg.DATALOADER.STD[0]
    image[:, 1, :, :] = (image[:, 1, :, :] -
                         cfg.DATALOADER.MEAN[1]) / cfg.DATALOADER.STD[1]
    image[:, 2, :, :] = (image[:, 2, :, :] -
                         cfg.DATALOADER.MEAN[2]) / cfg.DATALOADER.STD[2]
    return image


class GroupTasks:

    def __init__(self, tasks):
        self.tasks = tasks
        self.single_task_sim = 0

    def gen_task_combinations(self, similarity, rtn, index, path, path_dict):
        if index >= len(self.tasks):
            return

        for i in range(index, len(self.tasks)):
            cur_task = self.tasks[i]
            new_path = path
            new_dict = {k: v for k, v in path_dict.items()}

            # Building from a tree with two or more tasks...
            if new_path:
                new_dict[cur_task] = 0.
                for prev_task in path_dict:
                    new_dict[prev_task] += similarity[prev_task][cur_task]
                    new_dict[cur_task] += similarity[cur_task][prev_task]
                new_path = '{}|{}'.format(new_path, cur_task)
                rtn[new_path] = new_dict
            else:  # First element in a new-formed tree
                new_dict[cur_task] = 0.
                new_path = cur_task

            self.gen_task_combinations(
                similarity, rtn, i + 1, new_path, new_dict)

            if '|' not in new_path:
                new_dict[cur_task] = self.single_task_sim
                rtn[new_path] = new_dict

    def select_groups(self, rtn_tup, index, cur_group, best_group,
                      best_val, splits):
        # Check if this group covers all tasks.
        task_set = set()
        for group in cur_group:
            for task in group.split('|'):
                if task in task_set:
                    task_set.clear()
                    break
                task_set.add(task)
        if len(task_set) == len(self.tasks):
            best_tasks = {task: -1e6 for task in task_set}

            # Compute the per-task best scores for each task and
            # average them together.
            for group in cur_group:
                for task in cur_group[group]:
                    best_tasks[task] = max(
                        best_tasks[task], cur_group[group][task])
            group_avg = np.mean(list(best_tasks.values()))

            # Compare with the best grouping seen thus far.
            if group_avg > best_val[0]:
                best_val[0] = group_avg
                best_group.clear()
                for entry in cur_group:
                    best_group[entry] = cur_group[entry]

        # Base case.
        if len(cur_group.keys()) == splits:
            return

        # Back to combinatorics
        for i in range(index, len(rtn_tup)):
            selected_group, selected_dict = rtn_tup[i]

            new_group = {k: v for k, v in cur_group.items()}
            new_group[selected_group] = selected_dict

            if len(new_group.keys()) <= splits:
                self.select_groups(
                    rtn_tup, i + 1, new_group, best_group, best_val, splits)

    def call_generation(self, similarity):
        # rtn consists of all possible task combinations..
        # for each combination, each task has its average affinity with
        # all other tasks in the combination..
        rtn = {}
        self.gen_task_combinations(
            similarity, rtn=rtn, index=0, path='', path_dict={})
        # Normalize by the number of times the accuracy of any given
        # element has been summed.
        # i.e. (a,b,c) => [acc(a|b) + acc(a|c)]/2 + [acc(b|a) + acc(b|c)]/2 +
        # [acc(c|a) + acc(c|b)]/2
        for group in rtn:
            if '|' in group:
                for task in rtn[group]:
                    rtn[group][task] /= (len(group.split('|')) - 1)

        # 2^N - 1 combinations...
        assert (len(rtn.keys()) == 2 ** len(similarity.keys()) - 1)
        return [(key, val) for key, val in rtn.items()]

    def mat_to_dict(self, similarity):
        similarity_dict = {task: {} for task in self.tasks}
        for idx, task in enumerate(self.tasks):
            for sim, com_task in zip(similarity[idx], self.tasks):
                similarity_dict[task].update({com_task: sim})

        return similarity_dict

    def create_selection(self, similarity, min_similarity=0.5):
        similarity_dict = self.mat_to_dict(similarity)
        rtn_tup = self.call_generation(similarity_dict)
        unique_group = {}
        unique_val = [-100000000]
        for split in range(len(self.tasks)):
            selected_group = {}
            selected_val = [-100000000]
            self.select_groups(
                rtn_tup, index=0, cur_group={}, best_group=selected_group,
                best_val=selected_val, splits=split)
            is_unique = len(sum([g.split('|') for g in selected_group], [])
                            ) == len(self.tasks)
            unique_group = selected_group if is_unique else unique_group
            unique_val = selected_val if is_unique else unique_val
            if selected_val[0] >= min_similarity:
                break
        return unique_group, unique_val


def pearsonr(x, y, dim=-1):
    r"""Computes Pearson Correlation Coefficient across rows.
    https://github.com/audeering/audtorch/blob/master/audtorch/metrics/functional.py
    """
    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)
    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr.squeeze()


def cfg_node_to_dict(cfg_node, key_list):
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    else:
        cfg_dict = OrderedDict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_node_to_dict(v, key_list + [k])
        return cfg_dict
