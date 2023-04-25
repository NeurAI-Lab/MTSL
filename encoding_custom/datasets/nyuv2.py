import os
from collections import OrderedDict
import numpy as np
from PIL import Image

from encoding_custom.datasets.base_dataset import BaseDataset


class NYUv2Loader(BaseDataset):
    NUM_CLASSES = {'segment': 40, 'detect': 41, 'inst_seg': 41, 'inst_depth': 41}

    # INSTANCE_NAMES = ['__background__', 'bed', 'books', 'ceiling', 'chair', 'floor',
    #                   'furniture', 'objects', 'picture', 'sofa', 'table', 'tv', 'wall',
    #                   'window']

    INSTANCE_NAMES = ['__background__', 'wall', 'floor', 'cabinet', 'bed', 'chair',
                      'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
                      'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser',
                      'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books',
                      'refridgerator', 'television', 'paper', 'towel', 'shower curtain',
                      'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink',
                      'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture',
                      'otherprop']

    SEMANTIC_NAMES = ['__background__', 'wall', 'floor', 'cabinet', 'bed', 'chair',
                      'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
                      'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser',
                      'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books',
                      'refridgerator', 'television', 'paper', 'towel', 'shower curtain',
                      'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink',
                      'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture',
                      'otherprop']

    K = np.array([[5.8262448167737955e+02, 0, 3.1304475870804731e+02],
                  [0, 5.8269103270988637e+02, 2.3844389626620386e+02],
                  [0, 0, 1]])

    def __init__(self, root=os.path.expanduser('~/.encoding/data'),
                 split='train', mode=None, cfg=None, **kwargs):
        if split == 'minival':
            split = 'val'

        super(NYUv2Loader, self).__init__(
            root, split, cfg, mode, ann_path='annotations',
            ann_file_format='instances_%s_40.json', **kwargs)

        update_split = 'val' if split == 'test' else split
        (self.image_paths, self.mask_paths, self.depth_paths,
         self.normals_path) = get_paths(self.root, update_split)
        self.file_names = list(self.image_paths.keys())
        if len(self.image_paths) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" +
                               self.root + "\n")

        self.baseline = 1.
        self.focal_length = 1.

    def ann_check_hooks(self, obj):
        return True

    def get_detect_ids(self):
        ids = [0]
        for i in self.coco.cats.values():
            if i["name"] in self.INSTANCE_NAMES:
                ids.append(i["id"])
        return ids

    def get_segment_mapping(self):
        key = None
        segment_mapping = None

        return key, segment_mapping

    def depth_target_transform(self, disparity):
        return disparity

    def __getitem__(self, index):
        file_name = self.file_names[index]
        if not (self.split == 'train' or self.split == 'val'):
            return self.get_test_data(file_name)

        image = self.read_image(file_name)

        mask = self.seg_tar(file_name)
        if mask is not None:
            mask = mask.astype(np.int8) - 1

        depth = self.depth_tar(file_name)
        if depth is not None:
            # uint16 depth image are created using *1e3
            depth = depth / 1e3

        bboxes, labels, _, _, box_ids = self.inst_tar(
            file_name, disparity=depth, baseline=self.baseline,
            focal_length=self.focal_length, dist_scale=1.)

        sur_nor = None
        if 'sur_nor' in self.tasks:
            sur_nor = Image.open(self.normals_path[file_name])
            # normalize and -1 to 1 range...
            sur_nor = (np.array(sur_nor) / 255.) * 2 - 1

        if 'depth' in self.tasks:
            depth = self.depth_target_transform(depth)

        targets = OrderedDict(
            bboxes=bboxes, box_ids=box_ids, labels=labels,
            mask=mask, depth=depth, sur_nor=sur_nor)
        if self.transform:
            image, targets = self.transform(image, **targets)

        if 'segment' in self.tasks:
            mask = self.segment_target_transform(targets['mask'])

        if 'detect' not in self.tasks and self.is_det_req:
            bboxes = None
        targets = {'detect': bboxes, 'segment': mask,
                   'depth': targets['depth'],
                   'sur_nor': targets['sur_nor']}
        targets = {task: gt for task, gt in targets.items() if gt is not None}

        if 'ae' in self.tasks:
            targets.update({'ae': image})

        if self.split == 'val':
            return image, targets, index
        return image, targets


def get_paths(folder, split='train'):
    root_path = os.path.join(folder, 'image', split)
    image_paths = {}
    mask_paths = {}
    depth_paths = {}
    normals_paths = {}
    for file in os.listdir(root_path):
        img_path = os.path.join(root_path, file)
        image_paths.update({file: img_path})
        mask_paths.update({file: img_path.replace('image', 'seg40')})
        depth_paths.update({file: img_path.replace('image', 'depth')})
        normals_paths.update({file: img_path.replace('image', 'normal')})

    return image_paths, mask_paths, depth_paths, normals_paths
