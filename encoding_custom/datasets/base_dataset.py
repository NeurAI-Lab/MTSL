from abc import abstractmethod
import numpy as np
import os
from PIL import Image
import cv2
from pycocotools.coco import COCO
import torch
import torch.utils.data as data

from encoding_custom.datasets.transforms import build_transforms

__all__ = ['BaseDataset']


class BaseDataset(data.Dataset):

    def __init__(self, root, split, cfg, mode=None, crop_size=None,
                 ann_path=None, ann_file_format=None, has_inst_seg=True,
                 **kwargs):
        self.root = root
        self._split = split
        self.mode = mode
        self.crop_size = crop_size if crop_size is not None else [512, 512]
        self.tasks = [task for task, status in
                      dict(cfg.TASKS_DICT).items() if status]

        self.cfg = cfg
        self.transform = build_transforms(
            cfg, self.crop_size, self.mode == 'train', **kwargs)
        self.is_det_req = 'RandomSampleCrop' in cfg.DATALOADER.TRAIN_TRANSFORMS\
                          and mode == 'train'
        if not (mode == 'train' or mode == 'val'):
            return

        self._file_names = None
        self._image_paths = None
        self._mask_paths = None
        self._depth_paths = None

        self.coco = None
        if 'detect' in self.tasks or self.is_det_req:
            if ann_path is None:
                ann_path = cfg.DATALOADER.ANNOTATION_FOLDER
            if ann_file_format is None:
                ann_file_format = cfg.DATALOADER.ANN_FILE_FORMAT
            self.coco = COCO(os.path.join(root, ann_path, ann_file_format % split))

            self.image_ids = list(self.coco.imgs.keys())
            self.image_ids = sorted(self.image_ids)
            self.filename_to_id = dict()
            for i, ob in self.coco.imgs.items():
                self.filename_to_id[ob['file_name']] = ob['id']

            detect_ids = self.get_detect_ids()
            self.coco_id_to_contiguous_id = {coco_id: i for i, coco_id
                                             in enumerate(detect_ids)}
            self.contiguous_id_to_coco_id = {
                v: k for k, v in self.coco_id_to_contiguous_id.items()}

            self.add_area()

        self.key, self.segment_mapping = self.get_segment_mapping()

        self.has_inst_seg = has_inst_seg

    @property
    def image_size(self):
        return self.crop_size

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, value):
        assert type(value) is str, 'Dataset split should be string'
        self._split = value

    @property
    def file_names(self):
        return self._file_names

    @file_names.setter
    def file_names(self, value):
        assert type(value) is list, 'Expected a list of paths'
        self._file_names = value

    @property
    def image_paths(self):
        return self._image_paths

    @image_paths.setter
    def image_paths(self, value):
        assert type(value) is dict, 'filename to paths'
        self._image_paths = value

    @property
    def mask_paths(self):
        return self._mask_paths

    @mask_paths.setter
    def mask_paths(self, value):
        assert type(value) is dict, 'filename to paths'
        self._mask_paths = value

    @property
    def depth_paths(self):
        return self._depth_paths

    @depth_paths.setter
    def depth_paths(self, value):
        assert type(value) is dict, 'filename to paths'
        self._depth_paths = value

    @abstractmethod
    def get_detect_ids(self):
        pass

    @abstractmethod
    def get_segment_mapping(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def xywh2xyxy(box):
        x1, y1, w, h = box
        return [x1, y1, x1 + w, y1 + h]

    def get_img_info(self, index):
        file_name = self.file_names[index]
        image_id = self.filename_to_id[file_name]
        img_data = self.coco.imgs[image_id]
        return image_id, img_data

    @abstractmethod
    def ann_check_hooks(self, ann_obj):
        pass

    def get_test_data(self, file_name):
        image = self.read_image(file_name)
        original_image = image.copy()
        image, _ = self.transform(image)
        return image, original_image, file_name

    def read_image(self, file_name):
        image_path = self.image_paths[file_name]
        image = Image.open(image_path).convert('RGB')
        return np.array(image)

    def seg_tar(self, file_name):
        if 'segment' not in self.tasks:
            return None
        mask_path = self.mask_paths[file_name]
        mask = Image.open(mask_path)
        return np.array(mask)

    def depth_tar(self, file_name):
        if 'depth' not in self.tasks:
            return None
        return cv2.imread(self.depth_paths[file_name],
                          cv2.IMREAD_UNCHANGED).astype(np.float32)

    def inst_tar(self, file_name, disparity=None, baseline=None,
                 focal_length=None, dist_scale=256):
        if 'detect' not in self.tasks and not self.is_det_req:
            return None, None, None, None, None
        bboxes, labels, inst_masks = self.get_instances(file_name)
        box_ids = np.arange(0, len(bboxes))

        inst_depths = None
        if 'inst_depth' in self.tasks:
            inst_depths = self.get_inst_depth(
                bboxes, disparity, baseline, focal_length, dist_scale)

        return bboxes, labels, inst_masks, inst_depths, box_ids

    @staticmethod
    def get_inst_depth(boxes, disparity, baseline, focal_length,
                       dist_scale):
        assert disparity is not None, 'Invalid disparity..'
        assert baseline is not None, 'Invalid baseline..'
        assert focal_length is not None, 'Invalid focal length..'

        inst_depths = []
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = (int(box[0]), int(box[1]),
                                      int(box[2]), int(box[3]))
            xmin, ymin = max(0, xmin), max(0, ymin)
            if xmin == xmax:
                xmax = xmin + 1
            roi = disparity[ymin:ymax, xmin:xmax]
            dist = (baseline * focal_length * dist_scale) / max(
                np.median(roi) - 1, 1e-3)
            if dist > 1000:
                dist = 200
            dist = float(dist)
            inst_depths.append(dist)
        return np.array(inst_depths, dtype=np.float32)

    def get_instances(self, file_name):
        image_id = self.filename_to_id[file_name]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        loaded_anns = self.coco.loadAnns(ann_ids)

        bboxes, labels, inst_masks = [], [], []
        for obj in loaded_anns:
            if obj.get('iscrowd', 0) == 0 and obj.get('real_box', True)\
                    and self.ann_check_hooks(obj):
                bboxes.append(self.xywh2xyxy(obj["bbox"]))
                labels.append(self.coco_id_to_contiguous_id[obj["category_id"]])
                if self.has_inst_seg and 'inst_seg' in self.tasks:
                    inst_masks.append(self.coco.annToMask(obj))

        bboxes = np.array(bboxes, np.float32).reshape((-1, 4))
        labels = np.array(labels, np.int64).reshape((-1,))

        # remove invalid boxes
        keep = (bboxes[:, 3] > bboxes[:, 1]) & (bboxes[:, 2] > bboxes[:, 0])
        rets = [bboxes[keep], labels[keep], None]

        if self.has_inst_seg and 'inst_seg' in self.tasks:
            rets[2] = [inst_masks[idx] for idx, k in enumerate(keep) if k]

        return rets

    def add_area(self):
        for i, v in self.coco.anns.items():
            v['area'] = v['bbox'][2] * v['bbox'][3]

    def segment_target_transform(self, mask):
        mask = np.array(mask).astype('int32')
        if self.segment_mapping is not None:
            mask = self.segment_mask_to_contiguous(mask)
        return torch.from_numpy(mask).long()

    def segment_mask_to_contiguous(self, mask):
        values = np.unique(mask)
        for i in range(len(values)):
            assert (values[i] in self.segment_mapping)
        index = np.digitize(mask.ravel(), self.segment_mapping, right=True)
        return self.key[index].reshape(mask.shape)

    @abstractmethod
    def depth_target_transform(self, depth):
        pass
