from yacs.config import CfgNode as CN


def dense_options(_C):
    # --------------------------------------------------------------------------- #
    # Decoder Options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.DECODER = CN()
    _C.MODEL.DECODER.DECODER_TYPE = 'resnetv1'
    _C.MODEL.DECODER.ENABLE_SAM = False
    _C.MODEL.DECODER.USE_SAM_GAP = False
    _C.MODEL.DECODER.NUM_SAM_LAYERS = 8
    _C.MODEL.DECODER.OUTPLANES = 64
    _C.MODEL.DECODER.INIT_WEIGHTS = False

    # --------------------------------------------------------------------------- #
    # Segmentation Options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.SEG = CN()
    _C.MODEL.SEG.INPLANES = 64
    _C.MODEL.SEG.OUTPLANES = 64

    # --------------------------------------------------------------------------- #
    # Depth Options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.DEPTH = CN()
    _C.MODEL.DEPTH.INPLANES = 64
    _C.MODEL.DEPTH.OUTPLANES = 64
    _C.MODEL.DEPTH.ACTIVATION_FN = 'sigmoid'

    # --------------------------------------------------------------------------- #
    # Autoencoder Options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.AE = CN()
    _C.MODEL.AE.INPLANES = 64
    _C.MODEL.AE.OUTPLANES = 64
    _C.MODEL.AE.RECONST_LOSS = 'mse'

    return _C
