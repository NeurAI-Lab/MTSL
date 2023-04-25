import PIL.Image
import numpy as np
import cv2
import random
import torch
import logging
import torchvision.transforms as T
import albumentations as A
from albumentations.augmentations.bbox_utils import denormalize_bbox
import torch.nn.functional as F
from imagecorruptions import get_corruption_names, corrupt


class ImageCorruptions:
    def __init__(self, cfg, crop_size, corruption_name='gaussian_noise',
                 **kwargs):
        self.cfg = cfg
        self.crop_size = crop_size
        self.is_classify = kwargs.get('is_classify', True)
        self.severity = kwargs.get('severity', 1)
        if corruption_name is None:
            self.corruption_name = get_corruption_names()
        else:
            self.corruption_name = [corruption_name]

    def __call__(self, image, **targets):
        if self.is_classify:
            image = np.array(image)
        for corruption in self.corruption_name:
            image = corrupt(image, corruption_name=corruption,
                            severity=self.severity)
        if self.is_classify:
            return PIL.Image.fromarray(image)
        return image, targets


class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't support a lot of config settings
    and should only be used for production.
    """

    def __init__(self, cfg):
        super().__init__()
        self.crop_size = cfg.INPUT.IMAGE_SIZE
        self.mean = torch.as_tensor(cfg.DATALOADER.MEAN)[None, :, None, None]
        self.std = torch.as_tensor(cfg.DATALOADER.STD)[None, :, None, None]

    def forward(self, image):
        mean = self.mean.to(image.device)
        std = self.std.to(image.device)
        image = image / 255.
        image = image.permute(0, 3, 1, 2)
        image = F.interpolate(image, (self.crop_size[0], self.crop_size[1]),
                              mode='bilinear', align_corners=False)

        image.sub_(mean).div_(std)

        return image


class BaseSettings(object):
    def __init__(self, cfg, crop_size, **kwargs):
        self.cfg = cfg
        self.crop_size = crop_size
        self.kwargs = kwargs
        self.is_train = kwargs.get('is_train', True)
        self.img_scale = [tuple(cfg.INPUT.IMAGE_SIZE)]
        self.multiscale_mode = cfg.DATALOADER.MS_MULTISCALE_MODE
        self.ratio_range = cfg.DATALOADER.MS_RATIO_RANGE


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, **targets):
        for transform in self.transforms:
            image, targets = transform(image, **targets)
        return image, targets


class AddIgnoreRegions(BaseSettings):
    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)

    def __call__(self, image, **targets):
        if targets.get('sur_nor', None) is not None:
            sur_nor = targets['sur_nor']
            # Check areas with norm 0
            Nn = np.sqrt(
                sur_nor[:, :, 0] ** 2 + sur_nor[:, :, 1] ** 2 +
                sur_nor[:, :, 2] ** 2)

            targets['sur_nor'][Nn == 0, :] = 255.

        return image, targets


class AlbumentationsBase(object):
    def __init__(self, augmenter):
        self.augmenter = augmenter

    def get_params(self, image, targets):
        params = self.augmenter.get_params()
        params = self.augmenter.update_params(params, **{'image': image})

        return params

    def __call__(self, image, **targets):
        if random.random() < self.augmenter.p:
            params = self.get_params(image, targets)
            image = self.augmenter.apply(image, **params)

            if targets.get('bboxes', None) is not None and hasattr(
                    self.augmenter, 'apply_to_bboxes'):
                targets['bboxes'] = self.augmenter.apply_to_bboxes(
                    targets['bboxes'], **params)

            params['interpolation'] = cv2.INTER_NEAREST
            if targets.get('mask', None) is not None and hasattr(
                    self.augmenter, 'apply_to_mask'):
                targets['mask'] = self.augmenter.apply_to_mask(
                    targets['mask'], **params)
            if targets.get('inst_masks', None) is not None and hasattr(
                    self.augmenter, 'apply_to_masks'):
                targets['inst_masks'] = self.augmenter.apply_to_masks(
                    targets['inst_masks'], **params)
            if targets.get('depth', None) is not None and hasattr(
                    self.augmenter, 'apply_to_mask'):
                targets['depth'] = self.augmenter.apply_to_mask(
                    targets['depth'], **params)
            if targets.get('sur_nor', None) is not None and hasattr(
                    self.augmenter, 'apply_to_mask'):
                targets['sur_nor'] = self.augmenter.apply_to_mask(
                    targets['sur_nor'], **params)
            if targets.get('sem_cont', None) is not None and hasattr(
                    self.augmenter, 'apply_to_mask'):
                targets['sem_cont'] = self.augmenter.apply_to_mask(
                    targets['sem_cont'], **params)

        return image, targets


class Resize(AlbumentationsBase, BaseSettings):
    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)
        resizer = A.Resize(*self.crop_size, always_apply=True)
        AlbumentationsBase.__init__(self, resizer)


class ColorJitter(AlbumentationsBase, BaseSettings):
    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)
        jitterer = A.ColorJitter(p=0.5)
        AlbumentationsBase.__init__(self, jitterer)


class HorizontalFlip(AlbumentationsBase, BaseSettings):
    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)
        flipper = A.HorizontalFlip(p=0.5)
        AlbumentationsBase.__init__(self, flipper)


class VerticalFlip(AlbumentationsBase, BaseSettings):
    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)
        flipper = A.VerticalFlip(p=0.1)
        AlbumentationsBase.__init__(self, flipper)


class RandomCrop(AlbumentationsBase, BaseSettings):
    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)
        cropper = A.RandomCrop(*self.crop_size)
        AlbumentationsBase.__init__(self, cropper)


class RandomResizedCrop(AlbumentationsBase, BaseSettings):
    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)
        cropper = A.RandomResizedCrop(*self.crop_size)
        AlbumentationsBase.__init__(self, cropper)


class CropNonEmptyMaskIfExists(AlbumentationsBase, BaseSettings):
    """
    This augmentation is to include a random location of a
    random instance within the crop.
    """
    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)
        self.cropper = A.CropNonEmptyMaskIfExists(*self.crop_size)
        self.rand_crop = A.RandomCrop(*self.crop_size)
        AlbumentationsBase.__init__(self, self.cropper)

    def get_params(self, image, targets):
        params = self.augmenter.get_params()
        if targets.get('inst_masks', None) is not None:
            req_mask = random.choice(targets['inst_masks'])
        else:
            random_bbox = random.choice(targets['bboxes'])
            random_bbox = denormalize_bbox(random_bbox, *image.shape[:2])
            x0, y0, x1, y1 = [int(coord) for coord in random_bbox[:4]]
            req_mask = np.zeros(image.shape[:2])
            req_mask[y0: y1, x0: x1] = 1
        if random.random() < 0.8:
            self.augmenter = self.cropper
            params = self.augmenter.update_params(
                params, **{'mask': req_mask})
            params.update({"cols": image.shape[1], "rows": image.shape[0]})
        else:
            self.augmenter = self.rand_crop
            params = self.rand_crop.get_params()
            params = self.rand_crop.update_params(params, **{'image': image})

        return params


class PadIfNeeded(AlbumentationsBase, BaseSettings):
    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)
        padder = A.PadIfNeeded(*self.crop_size, p=1)
        AlbumentationsBase.__init__(self, padder)


class ShiftScaleRotate(AlbumentationsBase, BaseSettings):
    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)
        shift_limit = kwargs.get('shift_limit', 0)
        rotate_limit = kwargs.get('rotate_limit', 0)
        scale_limit = kwargs.get('scale_limit', [1.0, 1.2, 1.5])
        scaler = A.ShiftScaleRotate(
            shift_limit=shift_limit, scale_limit=scale_limit,
            rotate_limit=rotate_limit, border_mode=cv2.BORDER_CONSTANT,
            value=0, mask_value=0, p=1)
        AlbumentationsBase.__init__(self, scaler)


class Normalize(BaseSettings):
    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)
        self.mean = cfg.DATALOADER.MEAN
        self.std = cfg.DATALOADER.STD
        self.normalizer = T.Normalize(
            mean=cfg.DATALOADER.MEAN, std=cfg.DATALOADER.STD)

    def __call__(self, image, **targets):
        image = self.normalizer(image)

        return image, targets


class ResizeMultiScale(BaseSettings):
    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)
        self.resizer = Resize(cfg, crop_size, **kwargs)

        if len(self.ratio_range) > 0:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert self.multiscale_mode in ['value', 'range']

    @staticmethod
    def random_select(img_scales):
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale

    @staticmethod
    def random_sample(img_scales):
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        aspect_ratio = max(img_scale) / min(img_scale)
        max_idx = 1 if img_scale[1] > img_scale[0] else 0
        scale = [img_scale[0] * ratio, img_scale[1] * ratio]
        scale[max_idx] -= scale[1 - max_idx] % 1 * aspect_ratio
        scale = int(scale[0]), int(scale[1])
        return scale

    def _random_scale(self):
        if len(self.ratio_range) > 0:
            scale = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale = self.img_scale[0]
        elif self.multiscale_mode == 'range':
            scale = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        return scale

    def __call__(self, image, **targets):
        scale = self._random_scale()
        self.resizer.augmenter.height = scale[0]
        self.resizer.augmenter.width = scale[1]
        image, targets = self.resizer(image, **targets)
        return image, targets


class ToTensor(BaseSettings):
    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)
        self.to_tensor = T.ToTensor()

    def __call__(self, image, **targets):
        if targets.get('mask', None) is not None:
            targets['mask'] = torch.from_numpy(
                targets['mask'].astype(np.int64))

        if targets.get('depth', None) is not None:
            targets['depth'] = torch.from_numpy(
                targets['depth'].astype(np.float32))

        if targets.get('inst_masks', None) is not None:
            if len(targets['inst_masks']) > 0:
                targets['inst_masks'] = torch.from_numpy(
                    np.dstack(targets['inst_masks']).astype(np.int64))
            else:
                targets['inst_masks'] = torch.from_numpy(
                    np.empty(0).astype(np.int64))

        if targets.get('sur_nor', None) is not None:
            targets['sur_nor'] = torch.from_numpy(
                targets['sur_nor'].astype(np.float32))
            targets['sur_nor'] = targets['sur_nor'].permute(2, 0, 1)

        if targets.get('sem_cont', None) is not None:
            targets['sem_cont'] = torch.from_numpy(
                targets['sem_cont'].astype(np.int64))

        return self.to_tensor(image.astype(np.uint8)), targets


class ConvertFromInts(BaseSettings):
    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)

    def __call__(self, image, **targets):
        return image.astype(np.float32), targets


class PreProcessBoxes(BaseSettings):
    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)

        min_area = 16 if self.is_train else 0
        # visibility is calculated in relation to the area of bbox within
        # image.. 0.3 visibility would ensure that most of the
        # object is visible..
        min_visibility = 0.1 if self.is_train else 0
        bbox_params = A.BboxParams(
            format='pascal_voc', min_area=min_area,
            min_visibility=min_visibility, label_fields=['labels', 'box_ids'])
        self.bbox_process = A.BboxProcessor(bbox_params)

    def __call__(self, image, **targets):
        targets.update({'image': image})
        if targets.get('bboxes', None) is not None:
            self.bbox_process.preprocess(targets)
        targets.pop('image')
        return image, targets


class PostProcessBoxes(PreProcessBoxes):
    """
    Use this transform after all transforms which could
    translate instances outside image bounds..
    """
    def __init__(self, cfg, crop_size, **kwargs):
        PreProcessBoxes.__init__(self, cfg, crop_size, **kwargs)

    def __call__(self, image, **targets):
        if targets.get('bboxes', None) is not None:
            targets.update({'image': image})
            self.bbox_process.postprocess(targets)
            targets.pop('image')
            box_ids = targets['box_ids']
            if targets.get('inst_masks', None) is not None:
                targets['inst_masks'] = [targets['inst_masks'][idx]
                                         for idx in box_ids]
            if targets.get('inst_depths', None) is not None:
                targets['inst_depths'] = [targets['inst_depths'][idx]
                                          for idx in box_ids]

        return image, targets


class Expand(BaseSettings):
    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)

    def __call__(self, image, **targets):
        if np.random.randint(2):
            return image, targets

        boxes = targets.pop('bboxes', None)
        mask = targets.pop('mask', None)
        pixel_depth = targets.pop('depth', None)
        inst_masks = targets.pop('inst_masks', None)
        surface_normals = targets.pop('sur_nor', None)
        semantic_contours = targets.pop('sem_cont', None)

        height, width, c = image.shape
        ratio = random.uniform(1, 2)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), c),
            dtype=image.dtype)

        # make it zero instead of mean
        # expand_image[:, :, :] = self.mean
        expand_image[int(top):int(
            top + height), int(left):int(left + width)] = image
        image = expand_image

        # for mask
        if mask is not None:
            expand_mask = np.zeros(
                (int(height * ratio), int(width * ratio)),
                dtype=mask.dtype)
            # make it zero instead of mean
            # expand_image[:, :, :] = self.mean
            expand_mask[int(top):int(
                top + height), int(left):int(left + width)] = mask
            mask = expand_mask

        if inst_masks is not None:
            expand_inst_masks = []
            for inst in inst_masks:
                expand_inst = np.zeros(
                    (int(height * ratio), int(width * ratio)),
                    dtype=inst.dtype)
                # make it zero instead of mean
                # expand_image[:, :, :] = self.mean
                expand_inst[int(top):int(
                    top + height), int(left):int(left + width)] = inst
                expand_inst_masks.append(expand_inst)
            inst_masks = expand_inst_masks

        if pixel_depth is not None:
            expand_pixel_depth = np.zeros(
                (int(height * ratio), int(width * ratio)),
                dtype=pixel_depth.dtype) * -1
            # make it zero instead of mean
            # expand_image[:, :, :] = self.mean
            expand_pixel_depth[int(top):int(
                top + height), int(left):int(left + width)] = pixel_depth
            pixel_depth = expand_pixel_depth

        if surface_normals is not None:
            expand_surface_normals = np.zeros(
                (int(height * ratio), int(width * ratio), 3),
                dtype=surface_normals.dtype) * -1
            # make it zero instead of mean
            # expand_image[:, :, :] = self.mean
            expand_surface_normals[int(top):int(
                top + height), int(left):int(left + width), :] = surface_normals
            surface_normals = expand_surface_normals

        if semantic_contours is not None:
            expand_semantic_contours = np.zeros(
                (int(height * ratio), int(width * ratio)),
                dtype=mask.dtype)
            # make it zero instead of mean
            # expand_image[:, :, :] = self.mean
            expand_semantic_contours[int(top):int(
                top + height), int(left):int(left + width)] = semantic_contours
            semantic_contours = expand_semantic_contours

        if boxes is not None:
            boxes = boxes.copy()
            boxes[:, :2] += (int(left), int(top))
            boxes[:, 2:] += (int(left), int(top))

        targets.update({'bboxes': boxes, 'mask': mask, 'depth': pixel_depth,
                        'inst_masks': inst_masks, 'sur_nor': surface_normals,
                        'sem_cont': semantic_contours})
        return image, targets


class RandomSampleCrop(BaseSettings):

    def __init__(self, cfg, crop_size, **kwargs):
        BaseSettings.__init__(self, cfg, crop_size, **kwargs)
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, **targets):
        # guard against no boxes
        if targets['bboxes'] is not None and targets['bboxes'].shape[0] == 0:
            logging.warning('RandomSampleCrop is not performed as '
                            'bboxes are unavailable..')
            return image, targets

        boxes = targets.get('bboxes', None)
        labels = targets.get('labels', None)
        inst_depths = targets.get('inst_depths', None)

        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, targets

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image
                current_mask = targets.get('mask', None)
                current_pixel_depth = targets.get('depth', None)
                current_inst_masks = targets.get('inst_masks', None)
                current_surface_normals = targets.get('sur_nor', None)
                current_semantic_contours = targets.get('sem_cont', None)
                w = random.uniform(0.25 * width, width)
                h = int(1.0 * w * height / width + 0.5)

                # aspect ratio constraint b/t .5 & 2
                # if h / w < 0.5 or h / w > 2:
                #     continue

                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                # cut the crop from the mask
                if current_mask is not None:
                    current_mask = current_mask[rect[1]:rect[3], rect[0]:rect[2]]
                if current_inst_masks is not None:
                    current_inst_masks = [inst[rect[1]:rect[3], rect[0]:rect[2]]
                                          for inst in current_inst_masks]
                if current_pixel_depth is not None:
                    current_pixel_depth = current_pixel_depth[
                                          rect[1]:rect[3], rect[0]:rect[2]]
                if current_surface_normals is not None:
                    current_surface_normals = current_surface_normals[
                                          rect[1]:rect[3], rect[0]:rect[2], :]
                if current_semantic_contours is not None:
                    current_semantic_contours = current_semantic_contours[
                                                rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                boxs_mask = m1 * m2

                # have any valid boxes? try again if not
                if not boxs_mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[boxs_mask, :].copy()
                current_inst_depth = None
                if inst_depths is not None:
                    current_inst_depth = inst_depths[boxs_mask].copy()
                if current_inst_masks is not None:
                    current_inst_masks = [current_inst_masks[idx] for idx, b_m
                                          in enumerate(boxs_mask) if b_m]

                # take only matching gt labels
                current_labels = labels[boxs_mask]
                current_box_ids = np.arange(0, len(current_boxes))

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                current_targets = {'bboxes': current_boxes, 'labels': current_labels,
                                   'box_ids': current_box_ids, 'mask': current_mask,
                                   'depth': current_pixel_depth,
                                   'inst_depths': current_inst_depth,
                                   'inst_masks': current_inst_masks,
                                   'sur_nor': current_surface_normals,
                                   'sem_cont': current_semantic_contours}

                return current_image, current_targets


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
