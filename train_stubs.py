import copy
import os
import sys
import logging
import numpy as np
import yaml
import random
import time
import math
import json
import datetime
import torch.utils.data
import torch.backends.cudnn

from configs.defaults import _C as cfg
from encoding_custom.nn.model_fusion import AverageFusor, KnowledgeAmalgamation
from train import Trainer
from utilities import train_utils, generic_utils
from encoding_custom.losses.generic_losses import GenericLosses
from encoding_custom.nn.xai.compare_representations import CKA
from encoding_custom.nn.xai import xai_utils
from encoding_custom.nn.xai.explain import known_explainers


def parse_args():
    parser = train_utils.parse_args()
    parser.add_argument(
        '--trainer-name', default='DynamicProgressiveFusion', type=str, help='Train stub method')
    parser.add_argument(
        '--tse-ends', default=[3, 5, 7, 9, 13, 17, 25, 33, 41, 49], nargs="*", type=int,
        help='Epochs in the task specific phase')
    parser.add_argument(
        '--fusor', default='ka', type=str, help='Method for fusing weights')
    parser.add_argument(
        '--min-similarity', default=0.8, type=float, help='Minimum similarity for fusion')
    parser.add_argument(
        '--align-method', default='CKA', type=str, help='Alignment method.')
    parser.add_argument(
        '--amalgamate-epochs', default=[1, 1, 2, 2, 2, 2, 4, 4, 8, 8], nargs="*", type=int, help='Amalgamation epochs')
    parser.add_argument(
        '--amalgamate-method', default='coding', type=str, help='Amalgamate method.')
    parser.add_argument(
        '--amalgamate-loss', default='mse', type=str, help='Amalgamate loss.')
    parser.add_argument(
        '--copy-opt-state', action='store_true', help='copy optimizer state', default=False)
    parser.add_argument(
        '--avg-before-kd', action='store_true', help='average before knowledge distillation', default=False)
    parser.add_argument(
        '--cka-loss-w', default=0.2, type=float, help='CKA alignment loss weight.')

    return parser


class DynamicProgressiveFusion(Trainer):

    def __init__(self, args, logger):
        super(DynamicProgressiveFusion, self).__init__(args, logger)

        self.generic_losses = GenericLosses(cfg, ignore_cfg=True, loss_w=self.args.cka_loss_w)
        self.branching_info = copy.deepcopy(self.model.model_attributes.branching_info)
        self.current_task_groups = []
        for stem_name, stem_info in self.branching_info.items():
            group = [branch for branch in stem_info['branches'] if branch in self.tasks]
            if len(group) > 1:
                self.current_task_groups.append(group)
        self.args.compare_tasks = True
        self.explainer_args = getattr(cfg, 'EXPLAINER_ARGS', {})

        known_fusors = {'average': AverageFusor, 'ka': KnowledgeAmalgamation}
        self.fusor = known_fusors.get(self.args.fusor, None)

    def get_task_phase_end(self):
        end_epoch = self.args.epochs + 1
        for idx, tse_end in enumerate(self.args.tse_ends):
            if self.start_epoch < tse_end:
                end_epoch = tse_end
                break
        return end_epoch

    def loss_stub(self, loss, loss_dict, predictions, targets):
        if self.start_epoch >= max(self.args.tse_ends):
            return loss, loss_dict
        for group in self.current_task_groups:
            if len(group) <= 1 or group[0] not in self.model.task_to_first_stage.keys():
                continue
            group_preds = {}
            for task in group:
                name = self.model.task_to_first_stage[task]
                group_preds.update({name: predictions['activations'][name]})
            if self.args.align_method == 'CKA':
                group_preds = {'activations': group_preds}
                generic_loss, _ = self.generic_losses(group_preds, targets)
                alignment_loss = generic_loss['losses/feat_similarity']
            else:
                return loss, loss_dict

            loss = loss + alignment_loss
            group_name = '|'.join(group)
            loss_dict.update({f'losses/{group_name}_alignment_loss': alignment_loss})
        return loss, loss_dict

    def task_specific_phase(self, saver):
        logging.info('Started task-specific phase.')
        end_epoch = self.get_task_phase_end()
        for epoch in range(self.start_epoch, end_epoch):
            self.train_one_epoch(epoch)
            self.eval_and_save(epoch, saver)
            self.start_epoch += 1

    def update_branching_info(self, group):
        group_stem_name = '|'.join(group)
        stage_range = [0, 0]
        if group_stem_name in self.branching_info.keys():
            stage_range = self.branching_info[group_stem_name]['stage_range']
            stage_range[1] += 1
        else:
            for stem_name, stem_info in self.branching_info.items():
                # check if any of the task is the child of this stem...
                if group[0] in stem_info['branches']:
                    for t_g in group:
                        stem_info['branches'].remove(t_g)
                    stem_info['branches'].append(group_stem_name)
                    stem_end = stem_info['stage_range'][1]
                    stage_range = [stem_end + 1, stem_end + 1]
                    break
        self.branching_info.update({group_stem_name: dict(
            stage_range=stage_range, branches=group)})

    def fuse_layers(self):
        updated_model = generic_utils.get_model(
            cfg, self.args, self.norm_layer, self.tasks,
            branching_info=copy.deepcopy(self.branching_info))
        updated_model.load_state_dict(self.model.state_dict(), strict=False)
        fusor = self.fusor(self, updated_model)
        updated_model, new_optimizer = fusor.fuse(self.start_epoch)
        updated_model.to(self.device)
        self.model = updated_model
        self.optimizer = new_optimizer
        self.lr_scheduler.optimizer = new_optimizer
        if self.optimizer is None:
            self.init_optimizer()
            self.init_scheduler()
        else:
            self.lr_scheduler.optimizer = self.optimizer

    def fusion_phase(self):
        logging.info('Started fusion phase.')
        current_task_groups = copy.deepcopy(self.current_task_groups)
        self.current_task_groups.clear()
        ungrouped_tasks = []

        min_similarity = self.args.min_similarity

        explainer = CKA(
            self.args, cfg, self.model, self.tasks, self.dl_val,
            self.device, **self.explainer_args)
        grouping_info = {}
        for idx, group in enumerate(current_task_groups):
            if len(group) == 1:
                ungrouped_tasks.append(group)
                continue
            if group[0] not in self.model.task_to_first_stage.keys():
                logging.info(f'Grouping of {group} is complete.')
                continue
            select_groupings = generic_utils.GroupTasks(group)
            explainer.layer_names = [
                stage for task, stage in self.model.task_to_first_stage.items()
                if task in group]
            similarity = explainer.compare_across_tasks()
            grouping, _ = select_groupings.create_selection(
                similarity, min_similarity=min_similarity)
            grouping_info.update(grouping)
            explainer.features_x = None
            explainer.features_y = None

            for new_group in grouping.values():
                group_len = len(new_group.values())
                if group_len > 1 and sum(new_group.values()) / group_len > min_similarity:
                    tasks_in_group = list(new_group.keys())
                    for t_g in tasks_in_group:
                        group.remove(t_g)
                    self.current_task_groups.append(tasks_in_group)
                    self.update_branching_info(tasks_in_group)
            if len(group) > 0:
                ungrouped_tasks.append(group)

        grouping_info_file = os.path.join(
                self.root_dir, 'grouping_info_%03d.yaml' % self.start_epoch)
        grouping_dump = {}
        for key, val in grouping_info.items():
            val = {t: round(float(v), 4) for t, v in val.items()}
            grouping_dump.update({key: val})
        with open(grouping_info_file, 'w') as outfile:
            yaml.dump(grouping_dump, outfile, sort_keys=False)
        if len(self.current_task_groups) > 0:
            logging.info(f'New grouping: {self.current_task_groups}')
            self.fuse_layers()
            with open(self.args.config_file, 'r') as outfile:
                config = yaml.load(outfile, Loader=yaml.FullLoader)
            info_save_file = os.path.join(
                self.root_dir, 'branching_info_%03d.yaml' % self.start_epoch)
            config['DICT_CONFIG']['BRANCHING_INFO'] = json.loads(json.dumps(self.branching_info))
            with open(info_save_file, 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=None, sort_keys=False)

        logging.info(f'Ungrouped tasks: {ungrouped_tasks}')
        self.current_task_groups += ungrouped_tasks

    def train_and_evaluate(self):
        saver = train_utils.ModelSaver(cfg, self.root_dir, self.tasks, self.task_to_min_or_max)
        start_time = time.time()

        start_with_fusion = self.start_epoch > 1 and any([self.start_epoch % tse_end == 0
                                                          for tse_end in self.args.tse_ends])
        while self.start_epoch <= self.args.epochs:
            if not start_with_fusion:
                self.task_specific_phase(saver)

            mergeable_tasks = []
            stem_ends = []
            for stem_name, stem_info in self.branching_info.items():
                mergeable_tasks.append(list(set(self.tasks) & set(stem_info['branches'])))
                stem_ends.append(stem_info['stage_range'][1])

            do_fuse = False
            for m_tasks, s_ends in zip(mergeable_tasks, stem_ends):
                if len(m_tasks) > 1 and s_ends < self.model.model_attributes.total_stages:
                    do_fuse = True

            if do_fuse and self.start_epoch <= max(self.args.tse_ends):
                self.fusion_phase()
            else:
                logging.info('Fusion no longer possible.')
            start_with_fusion = False

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))


def main():
    # os.environ["RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "2"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6666"

    parser = parse_args()
    parser = xai_utils.parse_args(
        known_explainers.values(), parser=parser, add_common_args=False)
    args = parser.parse_args()
    train_utils.update_config_node(cfg, args)
    cfg.MISC.IS_ADV_TRAIN = True
    cfg.freeze()
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(args.seed)
        np.random.seed(args.seed)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    trainer = getattr(train_module, args.trainer_name)(args, logger)
    if args.eval_only:
        trainer.evaluate()
    else:
        trainer.train_and_evaluate()


if __name__ == "__main__":
    train_module = sys.modules[__name__]
    main()
