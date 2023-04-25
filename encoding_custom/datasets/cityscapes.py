import os
import numpy as np
import logging
import json
from collections import OrderedDict
import torch

from encoding_custom.datasets.base_dataset import BaseDataset


class CityscapesLoader(BaseDataset):
    NUM_CLASSES = {'segment': 19, 'detect': 9, 'inst_seg': 9, 'inst_depth': 9}

    INSTANCE_NAMES = ['__background__', 'person', 'rider', 'car',
                      'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    SEMANTIC_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                      'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                      'sky', 'person', 'rider', 'car', 'truck', 'bus',
                      'train', 'motorcycle', 'bicycle']

    def __init__(self, root=os.path.expanduser('~/.encoding/data'),
                 split='train', mode=None, cfg=None, **kwargs):

        if split == 'minival':
            split = 'val'
        super(CityscapesLoader, self).__init__(root, split, cfg, mode, **kwargs)

        if not (split == 'train' or split == 'val'):
            self.image_paths = get_city_pairs(root, split=split)
            self.file_names = list(self.image_paths.keys())
            if len(self.image_paths) == 0:
                raise RuntimeError(
                    "Found 0 images in subfolders of:" + self.root + "\n")
            return

        (self.image_paths, self.mask_paths, self.depth_paths,
         self.intrinsics_path, self.sem_conts_path) = get_city_pairs(
            self.root, self.split)
        self.file_names = list(self.image_paths.keys())
        assert (len(self.image_paths) == len(self.mask_paths))
        if len(self.image_paths) == 0:
            raise RuntimeError(
                "Found 0 images in subfolders of:" + self.root + "\n")

        self.min_depth = cfg.DATALOADER.MIN_DEPTH
        self.max_depth = cfg.DATALOADER.MAX_DEPTH
        self.baseline = 0.209313
        self.focal_length = 2262.52

    def ann_check_hooks(self, obj):
        return obj["category_id"] < self.NUM_CLASSES['detect']

    def get_detect_ids(self):
        ids = [0]
        for i in self.coco.cats.values():
            if i["name"] in self.INSTANCE_NAMES:
                ids.append(i["id"])
        return ids

    def get_segment_mapping(self):
        # Segmentation classes selected:
        # [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26,
        # 27, 28, 31, 32, 33]
        key = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1,
                        2, 3, 4, -1, -1, -1, 5, -1, 6, 7, 8, 9, 10, 11,
                        12, 13, 14, 15, -1, -1, 16, 17, 18])
        segment_mapping = np.array(
            range(-1, len(key) - 1)).astype('int32')

        return key, segment_mapping

    def depth_target_transform(self, disparity):
        disparity[disparity > 0] = ((disparity[disparity > 0] - 1) / 256.0)
        disparity[disparity > 0] = (self.baseline *
                                    self.focal_length) / disparity[disparity > 0]
        depth = disparity
        depth[depth < self.min_depth] = -1
        depth[depth > self.max_depth] = -1
        depth[depth > 0] = (depth[depth > 0] - self.min_depth) / (
                self.max_depth - self.min_depth)

        return depth

    def __getitem__(self, index):
        file_name = self.file_names[index]
        if not (self.split == 'train' or self.split == 'val'):
            return self.get_test_data(file_name)

        image = self.read_image(file_name)

        mask = self.seg_tar(file_name)
        depth = self.depth_tar(file_name)

        bboxes, labels, _, _, box_ids = self.inst_tar(
            file_name, disparity=depth, baseline=self.baseline,
            focal_length=self.focal_length)

        if 'depth' in self.tasks:
            depth = self.depth_target_transform(depth)

        targets = OrderedDict(
            bboxes=bboxes, box_ids=box_ids, labels=labels, mask=mask, depth=depth)
        if self.transform:
            image, targets = self.transform(image, **targets)

        if 'segment' in self.tasks:
            mask = self.segment_target_transform(targets['mask'])

        if 'detect' not in self.tasks and self.is_det_req:
            bboxes = None
        targets = {'detect': bboxes, 'segment': mask, 'depth': targets['depth']}
        targets = {task: gt for task, gt in targets.items() if gt is not None}

        if 'sur_nor' in self.tasks:
            intrinsics = get_intrisics(
                self.intrinsics_path[file_name], self.crop_size)
            targets.update({'intrinsics': intrinsics})

        if 'ae' in self.tasks:
            targets.update({'ae': image})

        if self.split == 'val':
            return image, targets, index
        return image, targets


def get_intrisics(path, crop_size):
    with open(path, 'r') as f:
        data = json.load(f)

    h_ratio = crop_size[0] / 1024
    w_ratio = crop_size[1] / 2048
    intrinsics = data['intrinsic']
    intrinsics = torch.tensor(
        [[intrinsics['fx'] * h_ratio, 0, intrinsics['u0'] * h_ratio],
         [0, intrinsics['fy'] * w_ratio, intrinsics['v0'] * w_ratio],
         [0, 0, 1]])
    return intrinsics


def get_image_paths(img_folder):
    img_paths = dict()
    for root, directories, files in os.walk(img_folder):
        for filename in files:
            if filename.endswith(".png") or filename.endswith(".jpg"):
                im_path = os.path.join(root, filename)

                if os.path.isfile(im_path):
                    img_paths[filename] = im_path
                else:
                    logging.info('cannot find the image:', im_path)
    logging.info('Found {} images in the folder {}'.format(
        len(img_paths), img_folder))
    return img_paths


def get_city_pairs(folder, split='train'):
    path_add = 'leftImg8bit/' + split
    if split == 'infer':
        path_add = ''
    img_paths = get_image_paths(os.path.join(folder, path_add))

    if split == 'train' or split == 'val':
        mask_paths = {}
        depth_paths = {}
        intrinsics_path = {}
        sem_conts_path = {}
        for filename, path in img_paths.items():
            depth_paths[filename] = path.replace('leftImg8bit', 'disparity')
            dir_name = os.path.dirname(path)
            dir_name = dir_name.replace('leftImg8bit', 'gtFine')
            base_name = os.path.basename(path)
            base_name = base_name.replace('leftImg8bit', 'gtFine_labelIds')
            mask_paths[filename] = os.path.join(dir_name, base_name)
            dir_name = dir_name.replace('gtFine', 'semantic_boundaries')
            base_name = base_name.replace('gtFine_labelIds', 'gtFine_edge')
            sem_conts_p = os.path.join(dir_name, base_name)
            sem_conts_p, _ = os.path.splitext(sem_conts_p)
            sem_conts_p = sem_conts_p + '.npy'
            sem_conts_path[filename] = sem_conts_p
            intrinsics_p = path.replace('leftImg8bit', 'camera')
            intrinsics_p, _ = os.path.splitext(intrinsics_p)
            intrinsics_p = intrinsics_p + '.json'
            intrinsics_path[filename] = intrinsics_p
        return img_paths, mask_paths, depth_paths, intrinsics_path, sem_conts_path
    else:
        return img_paths
