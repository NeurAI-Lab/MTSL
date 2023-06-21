from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

from encoding_custom.models.model_utils import *
from utilities import generic_utils


class ModelAttributes:

    def __init__(self, backbone, tasks, norm_layer, cfg, pretrained, **kwargs):
        self.tasks = tasks
        self.kwargs = kwargs

        self.branching_info = generic_utils.cfg_node_to_dict(cfg.BRANCHING_INFO, [])

        base_net = BaseNet(backbone, pretrained=pretrained,
                           norm_layer=norm_layer, **cfg.BACKBONE_KWARGS).backbone
        self.backbone_channels = base_net.feat_channels
        base_sequence = nn.Sequential(*list(base_net.children())[:-4])

        self.num_en_features = cfg.MODEL.ENCODER.NUM_EN_FEATURES
        self.en_feat_channels = self.backbone_channels[:self.num_en_features]
        en_layers = list(base_net.children())[-4:]

        self.stages = {f'en_stem_0': base_sequence}
        self.stages.update({f'en_stage_{idx + 1}': stage for idx, stage in enumerate(en_layers)})
        self.total_stages = len(en_layers)
        # 64 is the base stem feature channels...
        self.stage_to_channels = [64] + self.en_feat_channels


class ShatteredStages(nn.Module):
    def __init__(self, stage_range, model_attributes):
        super(ShatteredStages, self).__init__()

        self.num_en_features = model_attributes.num_en_features
        self.total_stages = model_attributes.total_stages
        for stage_name, stage in model_attributes.stages.items():
            if stage_range[0] <= int(stage_name.split('_')[-1]) <= stage_range[1]:
                self.add_module(f'{stage_name}', copy.deepcopy(stage))

    def forward(self, en_feats):
        named_children = self.named_children()
        name, stage_module = next(named_children)
        feats = en_feats
        while True:
            feats = stage_module(feats)
            name, stage_module = next(named_children, (None, None))
            if stage_module is None:
                break

        return feats


class MTSLArch(nn.Module):
    def __init__(self, backbone, tasks, norm_layer, cfg, pretrained=True,
                 branching_info=None, model_attributes=None, create_heads=True,
                 **kwargs):
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

        if create_heads:
            task_to_classes = {'segment': cfg.NUM_CLASSES.SEGMENT, 'depth': 1,
                               'sem_cont': 1, 'sur_nor': 3, 'ae': 3}
            for task in tasks:
                self.add_module(f'{task}_head', DeepLabHead(
                    cfg, task_to_classes[task], self.model_attributes.backbone_channels))

        for name, child in self.named_children():
            if name in tasks:
                self.task_to_first_stage.update(
                    {name: f'{name}.{list(child.named_children())[0][0]}'})

    @staticmethod
    def forward_single_stage(stage_module, stage_inputs, name):
        return stage_module(stage_inputs)

    def forward(self, x, targets=None, get_branch_to_inputs=False, do_interpolate=True,
                **kwargs):
        image_size = list(x.shape[2:])
        hooks, activations = [], None
        if len(self.task_to_first_stage) > 0:
            hooks, activations = generic_utils.register_forward_hooks(
                self, self.task_to_first_stage.values())

        results = OrderedDict()
        branch_to_inputs = {}
        for stem_name, stem_info in self.branching_info.items():
            stem_in = branch_to_inputs.get(stem_name, x)
            if stem_info['stage_range'][1] >= 0:
                stem_in = getattr(self, stem_name)(stem_in)
            for branch in stem_info['branches']:
                if branch in self.tasks and not get_branch_to_inputs:
                    task_feats = stem_in
                    if stem_info['stage_range'][1] < self.total_stages:
                        task_feats = getattr(self, branch)(stem_in)
                    results[branch] = getattr(self, branch + '_head')(task_feats)
                    if do_interpolate:
                        results[branch] = F.interpolate(
                            results[branch], image_size, mode='bilinear')
                    continue
                branch_to_inputs.update({branch: stem_in})

        for hook in hooks:
            hook.remove()

        if get_branch_to_inputs:
            return branch_to_inputs

        results.update({'activations': activations})
        return results


def get_mtsl_deeplab(backbone='resnet50', pretrained=True, **kwargs):
    tasks = kwargs.pop('tasks')
    norm_layer = kwargs.pop('norm_layer')
    cfg = kwargs.pop('cfg')
    model = MTSLArch(backbone, tasks, norm_layer, cfg, pretrained, **kwargs)

    return model
