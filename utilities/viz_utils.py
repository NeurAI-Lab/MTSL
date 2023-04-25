import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# colour map
label_colours_global = []
label_colours_global_det = []

city_seg_colors = np.asarray([[128, 64, 128], [232, 35, 244], [70, 70, 70],
                              [156, 102, 102], [153, 153, 190], [153, 153, 153],
                              [30, 170, 250], [0, 220, 220], [35, 142, 107],
                              [152, 251, 152], [180, 130, 70], [60, 20, 220],
                              [0, 0, 255], [142, 0, 0], [70, 0, 0], [100, 60, 0],
                              [100, 80, 0], [230,  0, 0], [32, 11, 119],
                              [255, 255, 255]], dtype=np.uint8)


def hex_to_rgb(hex_val):
    hex_val = hex_val.lstrip('#')
    hlen = len(hex_val)
    return tuple(int(hex_val[l:l + hlen // 3], 16)
                 for l in range(0, hlen, hlen // 3))


def rgb_to_rgb(hex_val):
    hlen = len(hex_val)
    return tuple(int(hex_val[l:l + hlen // 3], 16)
                 for l in range(0, hlen, hlen // 3))


colors = np.load('utilities/extra/colors.npy')
for i in colors:
    label_colours_global.append(hex_to_rgb(str(i)))
detcolors = np.load('utilities/extra/palette.npy')
for i in range(0, len(detcolors), 3):
    det = tuple((int(detcolors[i]), int(detcolors[i+1]), int(detcolors[i+2])))
    label_colours_global_det.append(det)


def decode_labels(mask, num_classes):
    """Decode batch of segmentation masks.
    Args:
      mask: result of inference after taking **argmax**.
      num_classes: number of classes to predict (including background).
    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    # init colours array
    colours = label_colours_global

    img = Image.new('RGB', (len(mask[0]), len(mask)))
    pixels = img.load()
    for j_, j in enumerate(mask[:, :]):
        for k_, k in enumerate(j):
            if k < num_classes:
                pixels[k_, j_] = colours[k]
    outputs = np.array(img)
    return outputs


def to_cv2_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    image = np.asarray(image, dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def seg_color_map(segment, seg_map=False, use_city_colors=True):
    if not seg_map:
        segment = segment.argmax(0).cpu().numpy()
    segment = np.array(segment, dtype=np.uint8)
    if use_city_colors:
        return city_seg_colors[segment]
    else:
        return colors[segment]


def depth_color_map(depth, cfg, depth_scale=None):
    if depth_scale is None:
        depth_scale = cfg.DATALOADER.MAX_DEPTH
    depth = depth.cpu().numpy()
    depth = np.squeeze(depth)
    # visualize disparity as in papers...
    depth = 1 - depth
    depth = to_depth_color_map(depth, depth_scale=depth_scale)
    return cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)


def to_depth_color_map(depth, depth_scale=1.):
    depth = (depth * depth_scale).astype(np.uint8)

    vmax = np.percentile(depth, 95)
    normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    depth = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)

    return depth
