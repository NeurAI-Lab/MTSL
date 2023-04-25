import logging
import torch
import numpy as np
import sys
import os
import time
import torch.utils.data
import torch.backends.cudnn
import cv2
from tqdm import tqdm
import torch.nn.functional as F
from ptflops import get_model_complexity_info

from configs.defaults import _C as cfg
from utilities.infer_utils import parse_args, run_inference, setup_infer_model
from utilities.generic_utils import back_transform
from utilities.viz_utils import to_depth_color_map, city_seg_colors, label_colours_global
from utilities.energy_meter import EnergyMeter


def single_image_infer(args):
    pass


def batched_inst_infer(args, device, tasks, model, dl_test):
    colors = np.asarray(label_colours_global, dtype=np.uint8)
    if args.dataset == 'uninet_cs':
        seg_colors = city_seg_colors
    else:
        seg_colors = colors

    start_time = time.time()
    with torch.no_grad():
        for idx, (image, original_image, image_name) in tqdm(
                enumerate(dl_test), desc='Running inference on images',
                total=len(dl_test)):
            image = image.to(device)
            predictions = model(image)

            # batch size is 1..
            original_image = original_image[0]
            original_image = original_image.cpu().numpy()
            original_image = np.asarray(original_image, dtype=np.uint8)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

            mask_size = original_image.shape[:2]
            if 'segment' in predictions.keys():
                segment = predictions['segment'][0]
                segment = F.interpolate(segment[None], size=mask_size,
                                        mode='nearest')

                segment = torch.squeeze(segment)
                segment = segment.argmax(0).cpu().numpy()
                segment = np.array(segment, dtype=np.uint8)
                segment = seg_colors[segment.copy()]
                original_image = np.hstack((original_image, segment))

            if 'sem_cont' in predictions.keys():
                pred_sem_cont = predictions['sem_cont'][0]
                # pred_sem_cont = torch.sigmoid(pred_sem_cont)
                pred_sem_cont = F.interpolate(pred_sem_cont[None], size=mask_size,
                                              mode='bilinear')
                pred_sem_cont.gt_(0)
                pred_sem_cont = pred_sem_cont.permute(0, 2, 3, 1).squeeze(0)

                pred_sem_cont = pred_sem_cont.cpu().numpy()
                if cfg.MISC.SEM_CONT_MULTICLASS:
                    sem_cont_map = np.zeros(pred_sem_cont.shape[:2])
                    for i in range(pred_sem_cont.shape[2]):
                        m = pred_sem_cont[:, :, i]
                        sem_cont_map[:, :] += (sem_cont_map == 0) * (m * (i + 1))
                    sem_cont_map = sem_cont_map - 1
                    sem_cont_map[sem_cont_map == -1] = cfg.NUM_CLASSES.SEGMENT
                else:
                    sem_cont_map = pred_sem_cont[:, :, 0]
                sem_cont_map = np.array(sem_cont_map, dtype=np.uint8)
                sem_cont_map = seg_colors[sem_cont_map]
                original_image = np.hstack((original_image, sem_cont_map))

            if 'depth' in predictions.keys():
                pred_depth = predictions['depth'][0, 0]
                pred_depth = F.interpolate(pred_depth[None, None], size=mask_size,
                                           mode='nearest')
                pred_depth = torch.squeeze(pred_depth).cpu().numpy()
                pred_depth = to_depth_color_map(
                    1 - pred_depth, depth_scale=cfg.DATALOADER.MAX_DEPTH)
                pred_depth = cv2.cvtColor(pred_depth, cv2.COLOR_RGB2BGR)
                original_image = np.hstack((original_image, pred_depth))

            if 'sur_nor' in predictions.keys():
                pred_sur_nor = predictions['sur_nor'][0]
                norm = torch.norm(pred_sur_nor, p=2, dim=0).unsqueeze(
                    dim=0) + 1e-12
                pred_sur_nor = pred_sur_nor.div(norm)
                pred_sur_nor = F.interpolate(
                    pred_sur_nor[None], size=mask_size, mode='bilinear')[0]
                pred_sur_nor = pred_sur_nor.permute(1, 2, 0)
                pred_sur_nor = ((pred_sur_nor + 1) / 2) * 255.
                pred_sur_nor = torch.squeeze(pred_sur_nor).cpu().numpy()
                pred_sur_nor = np.asarray(pred_sur_nor, dtype=np.uint8)
                original_image = np.hstack((original_image, pred_sur_nor))

            if 'ae' in predictions.keys():
                recon_img = predictions['ae']['reconst']
                recon_img = F.interpolate(
                    recon_img, size=mask_size, mode='bilinear')
                recon_img = back_transform(recon_img, cfg, scale=255)
                recon_img = recon_img.permute(0, 2, 3, 1).cpu().numpy()[0]
                recon_img = np.asarray(recon_img, dtype=np.uint8)
                recon_img = cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR)
                original_image = np.hstack((original_image, recon_img))

            if args.test_it > 0:
                cv2.imwrite(os.path.join(args.save_path, '%03d' % idx + '.png'),
                            original_image)
                if idx > args.test_it:
                    break
            else:
                cv2.namedWindow('viz', 0)
                cv2.imshow('viz', original_image)
                if cv2.waitKey(0) == ord('n'):
                    continue
                if cv2.waitKey(0) == ord('q'):
                    break

    if args.test_it > 0:
        total_time = time.time() - start_time
        logging.info(f'Turnaround time: {args.test_it / total_time}')


def get_inference_fps(model, data_loader, device, tasks, test_it=501):
    run_time = []
    data_sampler = iter(data_loader)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for i in range(test_it):
            try:
                images = next(data_sampler)[0]
            except StopIteration:
                batch_iterator = iter(data_loader)
                images = next(batch_iterator)[0]
            images = images.to(device)
            # device=None uses current device...
            torch.cuda.synchronize(device=None)

            start.record()
            predictions = model(images)
            run_inference(predictions, cfg, tasks)
            end.record()

            torch.cuda.synchronize(device=None)
            run_time.append(start.elapsed_time(end))

    run_time = run_time[1:]
    avg_run_time = np.mean(run_time)

    return 1000 / avg_run_time


def get_inference_fps_mmdet(model, data_loader, device, tasks):
    # https://github.com/open-mmlab/mmdetection/blob/d40e19b09b19dd3dd55627ecf0f8d2f0796a1a03/tools/benchmark.py
    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    for i, loaded in enumerate(data_loader):

        images = loaded[0]
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            images = images.to(device)
            predictions = model(images)
            run_inference(predictions, cfg, tasks)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time

        # using 1500 as test set only contains 1525 images..
        if (i + 1) == 1500:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            break

    return fps


def measure_fps(args, device, tasks, model, dl_test):
    fps = get_inference_fps(
        model, dl_test, device, tasks, test_it=501)
    # fps = get_inference_fps_mmdet(model, dl_test, device, tasks)
    logging.info(f'Measured FPS: {fps}')


def get_model_info(args, device, tasks, model, dl_test):
    macs, params = get_model_complexity_info(
        model, (3, ) + tuple(cfg.INPUT.IMAGE_SIZE), print_per_layer_stat=True)
    print(macs)
    print(params)


def measure_energy(args, device, tasks, model, dl_test, test_it=501):
    run_time = [0] * test_it
    images = torch.randn(1, 3, *cfg.INPUT.IMAGE_SIZE).cuda()
    with EnergyMeter() as em:
        for i in range(test_it):
            start = time.perf_counter()
            with torch.no_grad():
                output = model(images)
            torch.cuda.synchronize()  # wait for mm to finish
            run_time[i] = time.perf_counter() - start
            torch.cuda.synchronize()

        print(f"Total energy used check: {int(em.energy)} J")
        print(f'Average energy used: {em.energy / test_it} J')


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()

    model, tasks, dl_test, device = setup_infer_model(args, cfg)
    getattr(infer_module, args.function_name)(
        args, device, tasks, model, dl_test)


if __name__ == "__main__":
    infer_module = sys.modules[__name__]
    main()
