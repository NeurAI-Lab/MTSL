from collections import OrderedDict
import copy
import torch.nn as nn

from encoding_custom.backbones.base import BaseNet
from encoding_custom.models.model_utils import *
from utilities import generic_utils


class ModelAttributes:

    def __init__(self, backbone, tasks, norm_layer, cfg, pretrained, **kwargs):
        self.tasks = tasks
        self.kwargs = kwargs

        self.branching_info = generic_utils.cfg_node_to_dict(cfg.BRANCHING_INFO, [])

        base_net = BaseNet(backbone, pretrained=pretrained,
                           norm_layer=norm_layer, **cfg.BACKBONE_KWARGS).backbone
        backbone_channels = base_net.feat_channels
        base_sequence = nn.Sequential(*list(base_net.children())[:-4])

        uni_en, uni_en_channels = get_encoder(
            cfg.MODEL.ENCODER.ENCODER_TYPE, norm_layer, backbone_channels, cfg)
        en_feat_channels = backbone_channels + uni_en_channels
        self.num_en_features = cfg.MODEL.ENCODER.NUM_EN_FEATURES
        self.en_feat_channels = en_feat_channels[:self.num_en_features]
        en_layers = list(base_net.children())[-4:] + list(uni_en.children())

        decoder = get_decoder(
            cfg.MODEL.DECODER.DECODER_TYPE, norm_layer, self.en_feat_channels, cfg)

        self.stages = {f'en_stem_0': base_sequence}
        self.stages.update({f'en_stage_{idx + 1}': stage for idx, stage in enumerate(en_layers)})
        self.stages.update({f'de_stage_{self.num_en_features + idx + 1}': stage
                            for idx, stage in enumerate(decoder.blocks)})
        self.total_stages = len(en_layers) + len(decoder.blocks)
        # 64 is the base stem feature channels...
        self.stage_to_channels = [64] + self.en_feat_channels + [cfg.MODEL.DECODER.OUTPLANES * block.expansion
                                                                 for block in decoder.blocks]
        task_to_head_fn = {'segment': get_segment_head, 'depth': get_depth_head,
                           'sem_cont': get_sem_cont_head, 'sur_nor': get_sur_nor_head,
                           'ae': get_autoencoder_head}
        kwargs.update({'norm_layer': norm_layer})
        kwargs.update({'en_feat_channels': self.en_feat_channels})
        self.task_to_head = {}
        for task in tasks:
            self.task_to_head.update({task: task_to_head_fn[task](
                cfg, self.num_en_features - 1, **kwargs)})


class MTSLArch(nn.Module):
    def __init__(self, backbone, tasks, norm_layer, cfg, pretrained=True,
                 branching_info=None, model_attributes=None, **kwargs):
        super(MTSLArch, self).__init__()

        self.tasks = tasks
        self.task_to_first_stage = {}

        self.model_attributes = model_attributes
        if self.model_attributes is None:
            self.model_attributes = ModelAttributes(
                backbone, tasks, norm_layer, cfg, pretrained, **kwargs)

        self.num_en_features = self.model_attributes.num_en_features
        self.branching_info = self.model_attributes.branching_info
        if branching_info is not None:
            self.branching_info = branching_info

        self.total_stages = self.model_attributes.total_stages
        for stem_name, stem_info in self.branching_info.items():
            if stem_info['stage_range'][1] >= 0:
                self.add_module(f'{stem_name}', ShatteredStages(
                    stem_info['stage_range'], self.model_attributes))
            for branch in stem_info['branches']:
                if branch in tasks:
                    if stem_info['stage_range'][1] < self.total_stages:
                        self.add_module(f'{branch}', ShatteredStages(
                            [stem_info['stage_range'][1] + 1, self.total_stages],
                            self.model_attributes))

        for task in tasks:
            self.add_module(f'{task}_head', TaskHead(self.model_attributes, task))

        for name, child in self.named_children():
            if name in tasks:
                self.task_to_first_stage.update(
                    {name: f'{name}.{list(child.named_children())[0][0]}'})

    def forward_single_stage(self, stage_module, stage_inputs, name):
        en_feats, de_feats = stage_inputs
        x = en_feats[-1] if len(de_feats) == 0 else de_feats[-1]
        if 'de' in name:
            en_feats = en_feats[-self.num_en_features:]
            return stage_module([x, en_feats[
                self.total_stages - int(name.split('_')[-1])]])
        return stage_module(x)

    def forward(self, x, targets=None, get_branch_to_inputs=False):
        image_size = list(x.shape[2:])
        hooks, activations = [], None
        if len(self.task_to_first_stage) > 0:
            hooks, activations = generic_utils.register_forward_hooks(
                self, self.task_to_first_stage.values())

        results = OrderedDict()
        branch_to_inputs = {}
        for stem_name, stem_info in self.branching_info.items():
            stem_ins = branch_to_inputs.get(stem_name, [[x], []])
            stem_outs = [[], []]
            if stem_info['stage_range'][1] >= 0:
                stem_outs = getattr(self, stem_name)(*stem_ins)
            ins = [ins + outs for ins, outs in zip(stem_ins, stem_outs)]
            for branch in stem_info['branches']:
                if branch in self.tasks and not get_branch_to_inputs:
                    task_de_feats = []
                    if stem_info['stage_range'][1] < self.total_stages:
                        _, task_de_feats = getattr(self, branch)(*ins)
                    results[branch] = getattr(self, branch + '_head')(
                        image_size, ins[1] + task_de_feats)
                    continue
                branch_to_inputs.update({branch: ins})

        for hook in hooks:
            hook.remove()

        if get_branch_to_inputs:
            return branch_to_inputs

        results.update({'activations': activations})
        return results


class ShatteredStages(nn.Module):
    def __init__(self, stage_range, model_attributes):
        super(ShatteredStages, self).__init__()

        self.num_en_features = model_attributes.num_en_features
        self.total_stages = model_attributes.total_stages
        for stage_name, stage in model_attributes.stages.items():
            if stage_range[0] <= int(stage_name.split('_')[-1]) <= stage_range[1]:
                self.add_module(f'{stage_name}', copy.deepcopy(stage))

    def forward(self, shared_en_feats, shared_de_feats):
        shattered_en_feats = [shared_en_feats[-1]]
        named_children = self.named_children()
        name, stage_module = next(named_children)
        while True:
            if stage_module is None or name.split('_')[-3] != 'en':
                break
            feats = stage_module(shattered_en_feats[-1])
            shattered_en_feats.append(feats)
            name, stage_module = next(named_children, (None, None))
        shattered_en_feats.pop(0)

        shattered_de_feats = []
        en_feats = shared_en_feats + shattered_en_feats
        en_feats = en_feats[-self.num_en_features:]
        x = shared_de_feats[-1] if len(shared_de_feats) != 0 else en_feats[-1]
        while True:
            if stage_module is None:
                break
            x = stage_module([x, en_feats[self.total_stages - int(name.split('_')[-1])]])
            shattered_de_feats.append(x)
            name, stage_module = next(named_children, (None, None))

        return [shattered_en_feats, shattered_de_feats]


class TaskHead(nn.Module):
    def __init__(self, model_attributes, task):
        super(TaskHead, self).__init__()
        self.add_module(task + '_head', model_attributes.task_to_head[task])

    def forward(self, image_size, de_feats):
        return next(self.children())(de_feats, image_size)


def get_mtsl_arch(backbone='resnet50', pretrained=True, **kwargs):
    tasks = kwargs.pop('tasks')
    norm_layer = kwargs.pop('norm_layer')
    cfg = kwargs.pop('cfg')
    model = MTSLArch(backbone, tasks, norm_layer, cfg, pretrained, **kwargs)

    return model
