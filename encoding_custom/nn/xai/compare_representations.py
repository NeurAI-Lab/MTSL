from abc import abstractmethod
import copy

from tqdm import tqdm
import torch.backends.cudnn
import torch.utils.data

from encoding_custom.nn.xai import xai_utils
from encoding_custom.nn.similarity import CKASimilarity
from utilities import train_utils, generic_utils


class CompareRepresentation(xai_utils.BaseExplainer):
    """
    Use a default batch size of 8...
    """
    added_additional_args = False

    def __init__(self, args, cfg, model, tasks, dl, device, **kwargs):
        super(CompareRepresentation, self).__init__(
            args, cfg, model, tasks, dl, device, **kwargs)

        norm_layer = generic_utils.get_norm_layer(self.cfg, self.args)
        if args.single_task_model is not None and args.compare_models:
            self.single_model_seg_cls = args.single_model_seg_cls
            self.second_model = self.create_single_model(norm_layer)
            self.setup_second_model()
        elif args.multi_task_model is not None and args.compare_models:
            self.second_model = self.create_multi_model(norm_layer)
            self.setup_second_model()

        self.compare_models = args.compare_models
        self.compare_layers = args.compare_layers
        self.compare_tasks = args.compare_tasks

        self.metric_type = 'similarity'
        self.spatial_collapse = kwargs.get('spatial_collapse', True)

        base_name = 'encoder_decoder.%s_decoder.blocks.0'
        self.layer_names = [base_name % task for task in tasks]
        layer_names = kwargs.get('layer_names', None)
        if layer_names is not None:
            self.layer_names = layer_names

        self.features_x = None
        self.features_y = None

        self.plot = kwargs.get('plot', False)
        self.x_name = kwargs.get('x_name', '')
        self.y_name = kwargs.get('y_name', '')
        self.plot_save_path = kwargs.get('plot_save_path', None)
        self.title = kwargs.get('title', None)
        self.valid_tasks = kwargs.get('valid_tasks', self.tasks)
        task_to_letter = {'segment': 'S', 'depth': 'D',
                          'sem_cont': 'E', 'sur_nor': 'N', 'ae': 'A'}
        task_to_ticks = [r'$\mathcal{%s}$' % task_to_letter[task] for
                         task in self.valid_tasks]
        self.x_ticks = kwargs.get('x_ticks', task_to_ticks)
        self.y_ticks = kwargs.get('y_ticks', task_to_ticks)
        self.vmin = kwargs.get('vmin', 0)
        self.vmax = kwargs.get('vmax', 1)

        self.run_grouping = kwargs.get('run_grouping', False)
        if self.run_grouping:
            self.select_groupings = generic_utils.GroupTasks(self.valid_tasks)
            self.min_similarity = kwargs.get('min_similarity', 0.5)

    def setup_second_model(self):
        self.second_model.to(self.device)
        args = copy.deepcopy(self.args)
        args.resume = self.args.single_task_model
        train_utils.load_model(args, self.second_model, None)
        self.second_model.eval()

    def create_single_model(self, norm_layer):
        self.cfg.NUM_CLASSES.SEGMENT = self.single_model_seg_cls
        return generic_utils.get_model(
            self.cfg, self.args, norm_layer, [self.args.single_task])

    def create_multi_model(self, norm_layer):
        return generic_utils.get_model(
            self.cfg, self.args, norm_layer, self.tasks)

    def prepare_features(self, data, model=None):
        images = data[0]
        _, acts_out, _ = self.get_activations_predictions(images, model=model)
        acts_out.pop('images')
        dim = (2, 3) if self.spatial_collapse else (1, )

        return [torch.mean(acts_out[layer], dim=dim).flatten(1)
                for layer in self.layer_names]

    @staticmethod
    def gather_features(exist_features, new_features):
        if exist_features is None:
            exist_features = new_features
        else:
            exist_features = [torch.cat((exist_feat, new_feat))
                              for (exist_feat, new_feat) in
                              zip(exist_features, new_features)]

        return exist_features

    @abstractmethod
    def get_similarity(self):
        pass

    def compare_across_datasets(self):
        for batch_idx, (data_1, data_2) in tqdm(enumerate(
                zip(self.dl_1, self.dl_2)),
                desc=f'Measuring {self.metric_type} of same model '
                     f'with two datasets'):
            features_x = self.prepare_features(data_1)
            self.features_x = self.gather_features(self.features_x, features_x)
            features_y = self.prepare_features(data_2)
            self.features_y = self.gather_features(self.features_y, features_y)

        return self.get_similarity()

    def compare_across_models(self):
        for batch_idx, data in tqdm(
                enumerate(self.dl_1),
                desc=f'fMeasuring {self.metric_type} of two models'):
            features_x = self.prepare_features(data)
            self.features_x = self.gather_features(self.features_x, features_x)
            features_y = self.prepare_features(data, model=self.second_model)
            self.features_y = self.gather_features(self.features_y, features_y)

        return self.get_similarity()

    def compare_across_layers(self):
        layer_names_1 = self.layer_names[0]
        layer_names_2 = self.layer_names[1]
        self.layer_names = layer_names_1 + layer_names_2
        for batch_idx, data in tqdm(
                enumerate(self.dl_1),
                desc=f'Measuring {self.metric_type} across layers'):
            features = self.prepare_features(data)
            self.features_x = self.gather_features(
                self.features_x, features[:len(layer_names_1)])
            self.features_y = self.gather_features(
                self.features_y, features[len(layer_names_1):])

        return self.get_similarity()

    def compare_across_tasks(self):
        print(f'Total number of images: {len(self.dl_1.dataset)}')
        for batch_idx, data in tqdm(
                enumerate(self.dl_1),
                desc=f'Measuring {self.metric_type} across tasks'):
            features = self.prepare_features(data)
            self.features_x = self.gather_features(self.features_x, features)

        self.features_y = self.features_x
        return self.get_similarity()

    def explain(self):
        with torch.no_grad():
            if self.compare_models:
                assert hasattr(self, 'single_task_model')
                similarity = self.compare_across_models()
            elif self.compare_layers:
                similarity = self.compare_across_layers()
            elif self.compare_tasks:
                similarity = self.compare_across_tasks()
            else:
                similarity = self.compare_across_datasets()

            print(repr(similarity))
            if self.plot:
                xai_utils.plot_comparison_mat(
                    similarity, x_name=self.x_name, y_name=self.y_name,
                    save_path=self.plot_save_path + '.png'
                    if self.plot_save_path is not None else None,
                    x_ticks=self.x_ticks, y_ticks=self.y_ticks,
                    title=self.title, vmin=self.vmin, vmax=self.vmax)

            if self.run_grouping:
                grouping, value = self.select_groupings.create_selection(
                    similarity, min_similarity=self.min_similarity)
                print(f'Selected grouping: {grouping}')
                print(f'Value of selected grouping: {value}')

            return similarity

    @classmethod
    def additional_args(cls, parser):
        if not xai_utils.BaseExplainer.added_additional_args:
            parser = xai_utils.BaseExplainer.additional_args(parser)
        if CompareRepresentation.added_additional_args:
            return parser
        parser.add_argument(
            '--single-task-model', type=str, default=None,
            help='path containing the second model..')
        parser.add_argument(
            '--single-task', type=str, default='segment',
            help='single task to be used..')
        parser.add_argument(
            '--multi-task-model', type=str, default=None,
            help='path containing the second model which is multi task..')
        parser.add_argument(
            "--compare-models", default=False,
            help="compare across models; default is compare across datasets",
            action="store_true")
        parser.add_argument(
            "--compare-layers", default=False,
            help="compare across layers; default is compare across datasets",
            action="store_true")
        parser.add_argument(
            "--compare-tasks", default=False,
            help="compare across tasks; default is compare across datasets",
            action="store_true")
        parser.add_argument(
            '--single-model-seg-cls', type=int, default=19,
            help='path containing the second model..')
        CompareRepresentation.added_additional_args = True
        return parser


class CKA(CompareRepresentation, CKASimilarity):
    def __init__(self, args, cfg, model, tasks, dl_val, device, **kwargs):
        CompareRepresentation.__init__(
            self, args, cfg, model, tasks, dl_val, device, **kwargs)
        CKASimilarity.__init__(self, **kwargs)

        self.metric_type = 'CKA'
        self.metric_type += '_unbiased' if self.debiased else '_biased'
        self.metric_type += '_feature' if self.feature_based else '_sample'
        self.metric_type += '_spatial' if self.spatial_collapse\
            else '_cross_channel'

        self.kernel_type = kwargs.get('kernel_type', 'linear')

    def get_similarity(self):
        cka = torch.zeros((len(self.features_x), len(self.features_y)))

        for x_idx, feat_x in enumerate(self.features_x):
            for y_idx, feat_y in enumerate(self.features_y):
                cka[x_idx, y_idx] = self.calculate_similarity(
                    feat_x, feat_y, kernel_type=self.kernel_type,
                    replace_nan=False)

        cka[torch.eye(cka.shape[0]).byte()] = torch.nan_to_num(
            torch.diag(cka), nan=1.)
        return torch.nan_to_num(cka).numpy()
