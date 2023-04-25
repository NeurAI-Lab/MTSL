# --------------------------------------------------------------------------- #
# Dict config
# ---------------------------------------------------------------------------- #

BACKBONE_KWARGS = dict(
    # resnet args...
    dcn=None, dilate_only_last_layer=False, dilated=False, multi_grid=True, root='~/.encoding/models')

TRANSFORMS_KWARGS = dict()

TASKS_DICT = dict(segment=True, depth=True, sem_cont=True, sur_nor=True, ae=True)
TASK_TO_LOSS_NAME = dict(segment='default', depth='default',
                         sem_cont='default', sur_nor='default', ae='default')
TASK_TO_LOSS_ARGS = dict()
TASK_TO_LOSS_KWARGS = dict()
TASK_TO_LOSS_CALL_KWARGS = dict(segment=dict(ignore_index=-1))
TASK_TO_MIN_OR_MAX = dict(segment=1, depth=-1, sem_cont=-1, ae=-1)
LOSS_INIT_WEIGHTS = dict(segment_loss=1., depth_loss=1.)
LOSS_START_EPOCH = dict(detect_cls_loss=1, detect_reg_loss=1,
                        detect_centerness_loss=1, segment_loss=1,
                        depth_loss=1, inst_depth_l1_loss=1, inst_seg_loss=1)

LR_SCHEDULER_ARGS = dict()
OPTIMIZER_ARGS = dict()

# Dynamic Progressive Fusion
BRANCHING_INFO = dict()
