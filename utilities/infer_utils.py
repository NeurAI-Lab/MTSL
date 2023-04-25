import random
import numpy as np
import os
from PIL import Image
import torch
import torch.utils.data
import torch.backends.cudnn
from detectron2.structures import ImageList

from utilities import train_utils, generic_utils
from encoding_custom.datasets.transforms.transforms import ToTensor, \
    Normalize, Resize


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Multitask Inference')
    parser.add_argument(
        "--config-file", default="", metavar="FILE",
        help="path to config file", type=str)

    # dataset...
    parser.add_argument(
        '--dataset', default='uninet_cs', help='dataset')
    parser.add_argument(
        '--split', default='test',
        help='dataset split; use infer for inferring from folder')
    parser.add_argument(
        '--data-folder', type=str, default='data/',
        help='training dataset folder')

    # inference params...
    parser.add_argument(
        '--device', default='cuda', help='device')
    parser.add_argument(
        '--crop-size', type=int, default=[512, 512], nargs="*",
        help='crop image size')
    parser.add_argument(
        '-j', '--workers', default=16, type=int, metavar='N',
        help='number of data loading workers (default: 16)')
    parser.add_argument(
        '-b', '--batch-size', default=1, type=int)

    # misc...
    parser.add_argument(
        '--output-dir', default='./runs', help='path where to save')
    parser.add_argument(
        '--resume', default=None, help='resume from checkpoint')
    parser.add_argument(
        '--function-name', default="measure_fps", type=str,
        help='function to call')
    parser.add_argument(
        "--test-it", default=0, type=int,
        help="number of iterations for turnaround measure")
    parser.add_argument(
        '--save-path', default=None, type=str,
        help='img save path for turnaround measure')

    args = parser.parse_args()

    return args


def setup_infer_model(args, cfg, seed=0, num_workers=0, batch_size=1,
                      collate_fn=None, cfg_hook=None, freeze_cfg=True,
                      data_kwargs=None):
    # create attributes to use functions from train.py
    args.distributed = False
    args.backbone = False
    args.resume_after_suspend = False
    args.log_per_class_metrics = False
    args.local_rank = 0
    args.pretrained = False

    cfg = train_utils.update_config_node(cfg, args)
    if cfg_hook is not None:
        cfg_hook()
    if freeze_cfg:
        cfg.freeze()

    if seed >= 0:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)

    d_kwargs = {'crop_size': args.crop_size, 'root': args.data_folder,
                'cfg': cfg}
    if type(data_kwargs) is dict:
        data_kwargs.update(d_kwargs)
    else:
        data_kwargs = d_kwargs
    dataset = train_utils.get_multitask_dataset(
        args.dataset, split=args.split, **data_kwargs)
    sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    if collate_fn == 'train_collate':
        collate_fn = train_utils.BatchCollator(cfg, val=True)
    elif collate_fn == 'infer_collate':
        collate_fn = BatchCollator(cfg)
    dl = torch.utils.data.DataLoader(
        dataset, sampler=sampler, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True, batch_size=batch_size,
        drop_last=False)

    device = torch.device(args.device)

    norm_layer = generic_utils.get_norm_layer(cfg, args)
    tasks = [task for task, status in dict(cfg.TASKS_DICT).items() if status]
    model = generic_utils.get_model(cfg, args, norm_layer, tasks)
    model.to(device)
    if args.resume is not None:
        train_utils.load_model(args, model, None)
    model.eval()

    return model, tasks, dl, device


class BatchCollator:
    def __init__(self, cfg):
        self.size_divisibility = cfg.DATALOADER.SIZE_DIVISIBILITY

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = ImageList.from_tensors(
            transposed_batch[0], size_divisibility=self.size_divisibility)
        return images.tensor, transposed_batch[1], transposed_batch[2]


class ImageLoader(torch.utils.data.Dataset):

    def __init__(self, images_path, cfg, crop_size, max_images=None):
        self.files = []

        for root, directories, files in os.walk(images_path):
            for filename in files:
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    im_path = os.path.join(root, filename)
                    if os.path.isfile(im_path):
                        self.files.append(im_path)

        if max_images is not None:
            self.files = self.files[:max_images]

        self.resize = Resize(cfg, crop_size)
        self.to_tensor = ToTensor(cfg, cfg.INPUT.IMAGE_SIZE)
        self.normalize = Normalize(cfg, cfg.INPUT.IMAGE_SIZE)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert('RGB')
        image = np.array(image)
        original_image = image.copy()
        image = self.resize(image)[0]
        image, _ = self.normalize(self.to_tensor(image)[0])
        file_name = os.path.basename(self.files[index])
        return image, original_image, file_name

    def __len__(self):
        return len(self.files)
