import argparse
from abc import abstractmethod
from collections import OrderedDict
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt
import torch.backends.cudnn
import torch.utils.data

from utilities import infer_utils, generic_utils, train_utils


def parse_args(known_explainers, parser=None, add_common_args=True):
    if parser is None:
        parser = argparse.ArgumentParser(description='Explain networks')

    if add_common_args:
        parser = train_utils.add_common_args(parser)

    parser.add_argument(
        '--explainer-name', type=str,
        default='CKA', help='name of the explainer')

    for explainer in known_explainers:
        parser = explainer.additional_args(parser)

    return parser


def detach(tensor, requires_grad):
    if type(tensor) is torch.Tensor and not requires_grad:
        return tensor.detach()
    if type(tensor) is tuple:
        tensor = tuple(detach(ts, requires_grad) for ts in tensor)
    if type(tensor) is list:
        tensor = list([detach(ts, requires_grad) for ts in tensor])
    return tensor


def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def plot_comparison_mat(comparison_mat, x_name='', y_name='', x_ticks=None,
                        y_ticks=None, title=None, save_path=None,
                        vmin=0, vmax=1):
    # fig, ax = plt.subplots()
    im = plt.imshow(comparison_mat, vmin=vmin, vmax=vmax, origin='lower',
                    cmap='magma')
    add_colorbar(im)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    if x_ticks is not None:
        plt.xticks(list(range(comparison_mat.shape[0])), x_ticks)
    if y_ticks is not None:
        plt.yticks(list(range(comparison_mat.shape[1])), y_ticks)
    if title is not None:
        plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


class BaseExplainer:
    added_additional_args = False

    def __init__(self, args, cfg, model, tasks, dl, device, **kwargs):
        self.args = args
        self.cfg = cfg
        self.model = model
        self.tasks = tasks
        self.dl = dl
        self.device = device

        self.consider_tasks = kwargs.get(
            'consider_tasks', ['segment', 'detect'])
        self.layer_names = kwargs.get('layer_names', None)
        if self.layer_names is None:
            base_net = 'base_net.backbone.level'
            uni_en = 'encoder_decoder.uninet_encoder.stage'
            decoder = 'encoder_decoder.decoder.blocks.'
            self.layer_names = [base_net + '2', base_net + '3', base_net + '4',
                                base_net + '5', uni_en + '2', uni_en + '3',
                                decoder + '0', decoder + '1', decoder + '2',
                                decoder + '3', decoder + '4']

        max_images = kwargs.get('max_images', None)
        self.dl_1 = self.create_dls(args.data_folder_1, max_images=max_images)
        if args.data_folder_2 is not None:
            self.dl_2 = self.create_dls(
                args.data_folder_2, max_images=max_images)

    def create_dls(self, data_folder, max_images=None):
        dataset = infer_utils.ImageLoader(
            data_folder, self.cfg, self.args.crop_size, max_images=max_images)
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset, sampler=sampler, num_workers=self.args.workers,
            collate_fn=infer_utils.BatchCollator(self.cfg), pin_memory=True,
            batch_size=self.args.batch_size, drop_last=False)

    def get_layer_modules(self, model):
        layer_modules = []
        for layer_name in self.layer_names:
            layer_modules.append(generic_utils.get_module(model, layer_name))

        return layer_modules

    def register_forward_hooks(self, model, requires_grad):
        def get_activations(name, get_input=False):
            def forward_hook(module, inp, output=None):
                if get_input:
                    acts_inp[name] = detach(inp[0], requires_grad)
                else:
                    acts_out[name] = detach(output, requires_grad)
            return forward_hook

        acts_inp = OrderedDict()
        acts_out = OrderedDict()
        hooks = []
        layer_modules = self.get_layer_modules(model)
        for layer, layer_name in zip(layer_modules, self.layer_names):
            hooks.append(layer.register_forward_pre_hook(
                get_activations(layer_name, get_input=True)))
            hooks.append(layer.register_forward_hook(
                get_activations(layer_name)))

        return hooks, acts_inp, acts_out

    @staticmethod
    def remove_hooks(hooks):
        for hook in hooks:
            hook.remove()

    def add_additional_hooks(self, hooks):
        return hooks

    def pre_model_run(self, images, model, hooks):
        images = images.to(self.device)
        if not images.requires_grad:
            images.requires_grad = True
        if images.grad is not None:
            images.grad.data.fill_(0)
        images.retain_grad()

        hooks = self.add_additional_hooks(hooks)
        return images, model, hooks

    def post_model_run(self, images, model, hooks):
        self.remove_hooks(hooks)
        return images, model, hooks

    def get_activations_predictions(self, images, model=None,
                                    requires_grad=False):
        if model is None:
            model = self.model

        hooks, acts_inp, acts_out = self.register_forward_hooks(
            model, requires_grad)
        images, model, hooks = self.pre_model_run(images, model, hooks)
        predictions = model(images)
        images, model, hooks = self.post_model_run(images, model, hooks)

        acts_inp['images'] = images
        acts_out['images'] = images
        return acts_inp, acts_out, predictions

    @abstractmethod
    def explain(self):
        pass

    @classmethod
    def additional_args(cls, parser):
        if BaseExplainer.added_additional_args:
            return parser
        parser.add_argument(
            '--data-folder-1', type=str, default=None,
            help='path containing images from dataset 1.')
        parser.add_argument(
            '--data-folder-2', type=str, default=None,
            help='path containing images from dataset 2. If not provided,'
                 'one dataset alone will be used')
        BaseExplainer.added_additional_args = True
        return parser


class BaseEvaluator:

    @abstractmethod
    def evaluate(self, saliency, input_tensor, predictions, **kwargs):
        raise NotImplementedError

    def reduce_results(self, results):
        return results
