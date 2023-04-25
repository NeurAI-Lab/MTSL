from abc import abstractmethod
import torch
import numpy as np
from torch import nn

from configs.defaults import _C as cfg
from encoding_custom.optimizers import get_optimizer
from utilities import metric_utils, train_utils


class Fusor:

    def __init__(self, trainer, updated_model):
        self.args = trainer.args
        self.current_model = trainer.model
        self.updated_model = updated_model
        self.current_task_groups = trainer.current_task_groups
        self.writer = trainer.writer
        self.device = trainer.device
        self.dl_train = trainer.dl_train
        self.optimizer = trainer.optimizer

        self.new_optimizer = None
        if self.args.copy_opt_state:
            params_to_optimize = [
                {"params": [p for p in self.updated_model.parameters() if p.requires_grad]},
                {"params": [p for p in trainer.mtl_loss.parameters() if p.requires_grad]}]
            self.new_optimizer = get_optimizer(self.args, params_to_optimize, cfg)
            for idx, p_g in enumerate(self.optimizer.param_groups):
                self.new_optimizer.param_groups[idx]['lr'] = p_g['lr']

    def get_to_fuse(self, group):
        group_modules = [list(getattr(self.current_model, task).children())[0] for task in group]
        group_name = '|'.join(group)
        stage_range = self.updated_model.branching_info[group_name]['stage_range']
        fused_stage = stage_range[1] - stage_range[0]
        fusion_mod = getattr(self.updated_model, '|'.join(group)).named_children()
        fusion_mod = list(fusion_mod)[fused_stage]

        return [group_modules, fusion_mod[0], fusion_mod[1]]

    def update_opt_fused_params(self, group_modules, fusion_mod):
        # parameters are in order.
        group_params = [group.parameters() for group in group_modules]
        for f_param in fusion_mod.parameters():
            group_states = {}
            for g_param_iter, group_mod in zip(group_params, group_modules):
                g_param = next(g_param_iter)
                for key, value in self.optimizer.state[g_param].items():
                    group_states.setdefault(key, []).append(value)
                del self.optimizer.state[g_param]

            state = {}
            for key, value in group_states.items():
                if len(value) > 0:
                    if type(value[0]) is int:
                        state.update({key: int(np.mean(value))})
                    else:
                        state.update({key: torch.mean(torch.stack(value), dim=0)})
            self.new_optimizer.state.update({f_param: state})

    def update_opt_unfused_params(self):
        optimizer_items = iter(self.optimizer.state.items())
        for new_param in self.new_optimizer.param_groups[0]['params']:
            if new_param in self.new_optimizer.state.keys():
                continue
            param_not_matched = True
            while param_not_matched:
                try:
                    param, state = next(optimizer_items)
                except StopIteration:
                    return
                if torch.equal(new_param, param.to('cpu')):
                    self.new_optimizer.state.update({new_param: state})
                    param_not_matched = False

    @abstractmethod
    def fuse(self, start_epoch):
        pass


def average_parameter_fusion(fusion_mod, group_modules):
    group_mod_iters = [mod.parameters() for mod in group_modules]
    for name, param in fusion_mod.named_parameters():
        group_params = [next(it) for it in group_mod_iters]
        with torch.no_grad():
            param.copy_(torch.mean(torch.stack([g_p for g_p in group_params]), dim=0))


class AverageFusor(Fusor):
    def __init__(self, trainer, updated_model):
        super(AverageFusor, self).__init__(trainer, updated_model)

    def fuse(self, start_epoch):
        for group in self.current_task_groups:
            group_modules, _, fusion_mod = self.get_to_fuse(group)
            if self.args.copy_opt_state:
                self.update_opt_fused_params(group_modules, fusion_mod)
            average_parameter_fusion(fusion_mod, group_modules)

        if self.args.copy_opt_state:
            self.update_opt_unfused_params()
        return self.updated_model, self.new_optimizer


def get_amalg_loss(args):
    name_to_kd_loss = {'mse': nn.MSELoss}
    amalg_loss_fn = name_to_kd_loss.get(args.amalgamate_loss, None)
    if amalg_loss_fn is None:
        raise ValueError('Unknown amalgamate loss function')

    return amalg_loss_fn()


class ChannelCoding(nn.Module):
    def __init__(self, channels, num_teachers, args):
        super(ChannelCoding, self).__init__()
        self.amalg_loss = get_amalg_loss(args)

        amalg_mods = [nn.Sequential(nn.Linear(channels, channels), nn.ReLU(inplace=True),
                                    nn.Linear(channels, channels), nn.Sigmoid())
                      for _ in range(num_teachers)]
        self.amalg_mods = nn.ModuleList(amalg_mods)

    def forward(self, module_outs, fusion_mod_out):
        group_loss = 0
        projected_feats = [proj(torch.mean(fusion_mod_out, dim=(2, 3)))[:, :, None, None]
                           for proj in self.amalg_mods]
        for proj_feat, mod_out in zip(projected_feats, module_outs):
            group_loss = group_loss + self.amalg_loss(mod_out, proj_feat * fusion_mod_out)
        return group_loss


class KnowledgeAmalgamation(Fusor):
    def __init__(self, trainer, updated_model):
        super(KnowledgeAmalgamation, self).__init__(trainer, updated_model)
        known_amalgamators = {'coding': ChannelCoding}
        self.amalg_fn = known_amalgamators.get(self.args.amalgamate_method, None)
        if self.amalg_fn is None:
            raise ValueError('Unknown Amalgamator')

    def fuse(self, start_epoch):
        if len(self.args.amalgamate_epochs) > 1:
            current_ame = self.args.amalgamate_epochs.pop(0)
        else:
            current_ame = self.args.amalgamate_epochs[0]

        metric_logger = metric_utils.MetricLogger(delimiter="  ")
        header = f'Amalgamating knowledge'
        fusion_mappings = {'|'.join(group): self.get_to_fuse(group)
                           for group in self.current_task_groups if len(group) > 1}
        params_to_optimize = []
        for group_name, (group_modules, fusion_mod_name, fusion_mod) in fusion_mappings.items():
            if self.args.copy_opt_state:
                self.update_opt_fused_params(group_modules, fusion_mod)
            if self.args.avg_before_kd:
                average_parameter_fusion(fusion_mod, group_modules)
            params_to_optimize += list(fusion_mod.parameters())
            stage_num = int(fusion_mod_name.split('_')[-1])
            channels = self.updated_model.model_attributes.stage_to_channels[stage_num]
            amalg_mod = self.amalg_fn(channels, len(group_modules), self.args)
            amalg_mod.to(self.device)
            fusion_mappings[group_name].append(amalg_mod)
            params_to_optimize += list(amalg_mod.parameters())
            fusion_mod.to(self.device)
        params_to_optimize = [{"params": [p for p in params_to_optimize if p.requires_grad]}]
        optimizer = get_optimizer(self.args, params_to_optimize, cfg)
        if self.args.copy_opt_state:
            self.update_opt_unfused_params()

        for epoch in range(0, current_ame):
            for cnt, (images, targets) in enumerate(
                    metric_logger.log_every(self.dl_train, self.args.print_freq, header)):
                images, targets = train_utils.move_tensors_to_device(
                    images, targets, self.device)
                branch_to_inputs = self.current_model(images, get_branch_to_inputs=True)

                loss = 0
                for group_name, (group_modules, fusion_mod_name, fusion_mod,
                                 amalg_mod) in fusion_mappings.items():
                    # pick input to any task branch..
                    stage_inputs = branch_to_inputs[group_name.split('|')[0]]
                    module_outs = [self.current_model.forward_single_stage(
                        mod, stage_inputs, fusion_mod_name) for mod in group_modules]
                    fusion_mod_out = self.current_model.forward_single_stage(
                        fusion_mod, stage_inputs, fusion_mod_name)
                    group_loss = amalg_mod(module_outs, fusion_mod_out)
                    loss = loss + group_loss
                    metric_logger = train_utils.update_logger(
                        {f'losses/{group_name}_ka_loss': group_loss}, metric_logger)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.writer is not None:
                train_utils.log_meters_in_tb(
                    self.writer, metric_logger, start_epoch + epoch, 'train')
        for (_, _, _, amalg_mod) in fusion_mappings.values():
            del amalg_mod

        return self.updated_model, self.new_optimizer
