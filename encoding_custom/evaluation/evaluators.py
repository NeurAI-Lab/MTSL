from abc import abstractmethod
import torch
from torch import nn
import torch.distributed
import math

from utilities import dist_utils
from encoding_custom.losses.depth_losses import RMSE
from encoding_custom.losses.aux_losses import extract_semantic_contours,\
    extract_surface_normals, BalancedBinaryCrossEntropyLoss
from utilities.metric_utils import SmoothedValue


def get_mtl_evaluator(cfg, tasks, dataset_val, root_dir):
    mtl_evaluator = MultiTaskEvaluator(cfg)
    evaluators = {}
    if 'segment' in tasks:
        evaluators.update({'segment': SegmentEvaluator(cfg)})
    if 'depth' in tasks:
        evaluators.update({'depth': DepthEvaluator(cfg)})
    if 'sem_cont' in tasks:
        evaluators.update({'sem_cont': SemanticContourEvaluator(cfg)})
    if 'sur_nor' in tasks:
        evaluators.update({'sur_nor': SurfaceNormalsEvaluator(cfg, dataset_val)})
    if 'ae' in tasks:
        evaluators.update({'ae': AutoEncoderEvaluator(cfg)})

    mtl_evaluator.evaluators = evaluators

    return mtl_evaluator


class BaseEvaluator:

    def __init__(self, cfg):
        self.cfg = cfg
        self.log_per_cls_metrics = cfg.MISC.LOG_PER_CLASS_METRICS
        self.instance_names = cfg.INSTANCE_NAMES
        self.semantic_names = cfg.SEMANTIC_NAMES

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def process(self, targets, predictions, image_idxs):
        pass

    @abstractmethod
    def reduce_from_all_processes(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class MultiTaskEvaluator(BaseEvaluator):

    def __init__(self, cfg):
        # based on detectron2 DatasetEvaluators
        super(MultiTaskEvaluator, self).__init__(cfg)
        self.aux_tasks = cfg.MISC.AUX_TASKS
        self._evaluators = {}

    @property
    def evaluators(self):
        return self._evaluators

    @evaluators.setter
    def evaluators(self, evaluators_dict):
        self._evaluators = evaluators_dict

    def reset(self):
        for evaluator in self.evaluators.values():
            evaluator.reset()

    def process(self, targets, predictions, image_idxs):
        for task, evaluator in self.evaluators.items():
            if task in self.aux_tasks or task is 'generic':
                evaluator.process(targets, predictions, image_idxs)
            else:
                evaluator.process(targets[task], predictions[task], image_idxs)

    def evaluate(self):
        self.reduce_from_all_processes()
        results = {}
        for evaluator in self.evaluators.values():
            result = evaluator.evaluate()
            if dist_utils.is_main_process() and result is not None:
                for k, v in result.items():
                    assert (k not in results), \
                        "Different evaluators produce results with " \
                        "the same key {}".format(k)
                    results[k] = v
        return results

    def reduce_from_all_processes(self):
        for evaluator in self.evaluators.values():
            evaluator.reduce_from_all_processes()


class SegmentEvaluator(BaseEvaluator):

    def __init__(self, cfg, num_classes=None):
        super(SegmentEvaluator, self).__init__(cfg)
        if num_classes is not None:
            self.num_classes = num_classes
        else:
            self.num_classes = cfg.NUM_CLASSES.SEGMENT
        self.confusion_matrix = None

    def reset(self):
        self.confusion_matrix.zero_()

    def _gen_confusion_mat(self, prediction, target, mask=None):
        #https://discuss.pytorch.org/t/more-classes-slower-confusion-matrix-bincount/123591
        if mask is not None:
            temp = self.num_classes * target[mask] + prediction[mask]
        else:
            temp = self.num_classes * target + prediction

        i = 0
        conf_mat = torch.zeros(
            self.num_classes**2, dtype=torch.int32, device=target.device)
        while i < self.num_classes**2:
            t_mask = temp >= i
            t_mask &= temp < i + 999
            minlength = self.num_classes**2-i if i + 999 > self.num_classes**2 else 999
            conf_mat[i:i+999] = torch.bincount(
                temp[t_mask] - i, minlength=minlength)
            i += 999

        return conf_mat.reshape(self.num_classes, self.num_classes)

    def process(self, targets, predictions, image_idxs):
        # tars_flat = targets.flatten()
        # preds_flat = predictions.argmax(1).flatten()
        n = self.num_classes
        if self.confusion_matrix is None:
            self.confusion_matrix = torch.zeros(
                (n, n), dtype=torch.int64, device=targets.device)
        confusion_matrix = self._gen_confusion_mat(predictions.argmax(1), targets)
        self.confusion_matrix += confusion_matrix
        # with torch.no_grad():
        #     k = (tars_flat >= 0) & (tars_flat < n)
        #     inds = n * tars_flat[k].to(torch.int64) + preds_flat[k]
        #     self.confusion_matrix += torch.bincount(
        #         inds, minlength=n ** 2).reshape(n, n)

    def compute(self):
        h = self.confusion_matrix.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.confusion_matrix)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return ("global correct: {:.1f}\n average row correct: {}\n"
                "IoU: {}\n mean IoU: {:.1f}").format(
            acc_global.item() * 100,
            ["{:.1f}".format(i) for i in (acc * 100).tolist()],
            ["{:.1f}".format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100)

    def evaluate(self):
        acc_global, acc, iu = self.compute()
        acc_global = acc_global.item()
        mean_iou = iu.mean().item() * 100
        class_iu = (iu * 100).tolist()
        result = {'metrics/segment_pixel_acc': acc_global,
                  'key_metrics/segment_MIoU': mean_iou}
        if self.log_per_cls_metrics:
            for sem, c_iu in zip(self.semantic_names, class_iu):
                result.update({f'classwise/iou_{sem}': c_iu})
        return result


class DepthEvaluator(BaseEvaluator):

    def __init__(self, cfg):
        super(DepthEvaluator, self).__init__(cfg)
        self.min_depth = cfg.DATALOADER.MIN_DEPTH
        self.max_depth = cfg.DATALOADER.MAX_DEPTH
        self.rmse_fn = RMSE(cfg)
        self.rmse = SmoothedValue()
        self.abs_rel = SmoothedValue()
        self.abs_err = SmoothedValue()
        self.sq_rel = SmoothedValue()
        self.rmse_log = SmoothedValue()
        self.a1 = SmoothedValue()
        self.a2 = SmoothedValue()
        self.a3 = SmoothedValue()

    def reset(self):
        pass

    def process(self, targets, predictions, image_idxs):
        # scale back to range 0 to max_depth from 0 to 1..
        targets = self.max_depth * targets
        predictions = self.max_depth * predictions
        rmse_value = self.rmse_fn(predictions, targets)
        self.rmse.update(rmse_value)

        mask = torch.where(targets >= 0)
        predictions = torch.squeeze(predictions, dim=1)
        (abs_rel, abs_err, sq_rel, rmse_log, a1, a2, a3) = compute_depth_errors(
            targets[mask], predictions[mask])
        self.abs_rel.update(abs_rel)
        self.abs_err.update(abs_err)
        self.sq_rel.update(sq_rel)
        self.rmse_log.update(rmse_log)
        self.a1.update(a1)
        self.a2.update(a2)
        self.a3.update(a3)

    def reduce_from_all_processes(self):
        pass

    def evaluate(self):
        return {'key_metrics/depth_rmse': self.rmse.global_avg,
                'metrics/depth_abs_rel': self.abs_rel.global_avg,
                'metrics/depth_abs_err': self.abs_err.global_avg,
                'metrics/depth_sq_rel': self.sq_rel.global_avg,
                'metrics/depth_rmse_log': self.rmse_log.global_avg,
                'metrics/depth_a1': self.a1.global_avg,
                'metrics/depth_a2': self.a2.global_avg,
                'metrics/depth_a3': self.a3.global_avg}


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.where((gt / pred) > (pred / gt), (gt / pred), (pred / gt))
    a1 = thresh[(thresh < 1.25)].mean()
    a2 = thresh[(thresh < 1.25 ** 2)].mean()
    a3 = thresh[(thresh < 1.25 ** 3)].mean()

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    abs_err = torch.mean(torch.abs(gt - pred))

    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    return abs_rel, abs_err, sq_rel, rmse_log, a1, a2, a3


class SemanticContourEvaluator(BaseEvaluator):

    def __init__(self, cfg):
        super(SemanticContourEvaluator, self).__init__(cfg)
        self.sem_consist_acc = SmoothedValue()
        self.num_seg_classes = cfg.NUM_CLASSES.SEGMENT
        self.multiclass = cfg.MISC.SEM_CONT_MULTICLASS
        self.ce_loss = BalancedBinaryCrossEntropyLoss(cfg)
        self.pixel_accuracy = SmoothedValue()
        self.sim_err = SmoothedValue()
        self.sem_consist_err = SmoothedValue()

    def reset(self):
        pass

    def pixel_acc(self, preds, tars):
        if tars.ndim == 4:
            tars = tars.argmax(1)
        sem_cont_target = extract_semantic_contours(
            tars, self.num_seg_classes, multi_class=self.multiclass)
        matches = preds.argmax(1) == sem_cont_target
        return torch.sum(matches) / matches.nelement()

    def process(self, targets, predictions, image_idxs):
        acc = self.pixel_acc(predictions['sem_cont'], targets['segment'])
        consist = self.pixel_acc(predictions['sem_cont'], predictions['segment'])
        self.pixel_accuracy.update(acc)
        self.sem_consist_acc.update(consist)

        consistency_loss = self.ce_loss(predictions['sem_cont'], predictions)
        self.sem_consist_err.update(consistency_loss)
        ce_loss = self.ce_loss(predictions['sem_cont'], targets)
        self.sim_err.update(ce_loss)

    def reduce_from_all_processes(self):
        pass

    def evaluate(self):
        return {'key_metrics/sem_cont_bce_err': self.sim_err.global_avg,
                'metrics/sem_cont_pixel_accuracy': self.pixel_accuracy.global_avg,
                'metrics/semantic_consistency_err': self.sem_consist_err.global_avg,
                'metrics/semantic_consistency_acc': self.sem_consist_acc.global_avg}


class SurfaceNormalsEvaluator(BaseEvaluator):

    def __init__(self, cfg, dataset_val):
        super(SurfaceNormalsEvaluator, self).__init__(cfg)
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.geo_consist = SmoothedValue()
        self.cos_sim = SmoothedValue()
        self.nor_mean = SmoothedValue()
        self.nor_rmse = SmoothedValue()
        self.nor_11_25 = SmoothedValue()
        self.nor_22_5 = SmoothedValue()
        self.nor_30 = SmoothedValue()
        self.dataset_val = dataset_val

    def reset(self):
        pass

    def process(self, targets, predictions, image_idxs):
        intrinsics = targets.get('intrinsics', None)
        if intrinsics is None and hasattr(self.dataset_val, 'K'):
            intrinsics = self.dataset_val.K.copy()
            intrinsics = torch.from_numpy(intrinsics)[None]
            intrinsics = intrinsics.repeat(predictions['sur_nor'].size(0), 1, 1)
            intrinsics = intrinsics.to(predictions['sur_nor'].device)

        if targets.get('sur_nor', None) is not None:
            target_nor = targets['sur_nor']
        else:
            target_nor = extract_surface_normals(
                targets['depth'][:, None, :, :], intrinsics)

        l2_norm = torch.sqrt(
                target_nor[:, 0, ...] ** 2 + target_nor[:, 1, ...] ** 2 +
                target_nor[:, 2, ...] ** 2)
        valid_mask = l2_norm != 0
        invalid_mask = 1 - valid_mask.long()
        pred = predictions['sur_nor'].permute(0, 2, 3, 1)
        pred[invalid_mask == 1, :] = 0
        pred = pred.permute(0, 3, 1, 2)

        if intrinsics is not None:
            convert_nor = extract_surface_normals(
                predictions['depth'], intrinsics)
            consistency_loss = self.cosine_similarity(pred, convert_nor)
            self.geo_consist.update(consistency_loss.mean())

        cos_sim = self.cosine_similarity(pred, target_nor)
        self.cos_sim.update(cos_sim.mean())

        deg_diff = (180 / math.pi) * (torch.acos(torch.clamp(torch.sum(
            pred * target_nor, 1), min=-1, max=1)))
        deg_diff = torch.masked_select(deg_diff, valid_mask)

        nor_mean, nor_rmse, nor_11_25, nor_22_5, nor_30 = compute_normal_errors(
            deg_diff)
        self.nor_mean.update(nor_mean)
        self.nor_rmse.update(nor_rmse)
        self.nor_11_25.update(nor_11_25)
        self.nor_22_5.update(nor_22_5)
        self.nor_30.update(nor_30)

    def reduce_from_all_processes(self):
        pass

    def evaluate(self):
        res = {'metrics/mean': self.nor_mean.global_avg,
               'metrics/rmse': self.nor_rmse.global_avg,
               'metrics/11.25': self.nor_11_25.global_avg,
               'metrics/22.5': self.nor_22_5.global_avg,
               'metrics/30': self.nor_30.global_avg,
               'key_metrics/sur_nor_cosine_similarity': self.cos_sim.global_avg}
        if self.geo_consist.count > 0:
            res.update({'metrics/geometric_consistency': self.geo_consist.global_avg})
        return res


def compute_normal_errors(deg_diff):
    n = deg_diff.numel()
    nor_mean = torch.sum(deg_diff).item() / n
    nor_rmse = torch.sum(torch.sqrt(torch.pow(deg_diff, 2))).item() / n
    nor_11_25 = torch.sum((deg_diff < 11.25).float()).item() * 100 / n
    nor_22_5 = torch.sum((deg_diff < 22.5).float()).item() * 100 / n
    nor_30 = torch.sum((deg_diff < 30).float()).item() * 100 / n

    return nor_mean, nor_rmse, nor_11_25, nor_22_5, nor_30


class AutoEncoderEvaluator(BaseEvaluator):

    def __init__(self, cfg):
        super(AutoEncoderEvaluator, self).__init__(cfg)
        self.ae_mse = SmoothedValue()
        self.ae_loss = nn.MSELoss()

    def reset(self):
        pass

    def process(self, targets, predictions, image_idxs):
        ae_mse_loss = self.ae_loss(predictions['reconst'], targets)
        self.ae_mse.update(ae_mse_loss)

    def reduce_from_all_processes(self):
        pass

    def evaluate(self):
        metrics = {'key_metrics/ae_mse': self.ae_mse.global_avg}
        return metrics
