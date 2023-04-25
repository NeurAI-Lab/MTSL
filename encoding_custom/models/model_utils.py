from ..encoders_decoders.decoder import Decoder, DecoderBlock
from ..encoders_decoders.uninet_encoder import resnet_encoder
from encoding_custom.heads.segment.seg_head import SegHead
from ..heads.depth_head import DepthHead
from ..heads.aux_heads import SemanticContourHead, SurfaceNormalHead
from ..heads.autoencoder import AutoEncoderHead


def get_encoder(encoder_type, norm_layer, backbone_feat_channels, cfg):
    models = {'resnet': resnet_encoder}
    model_args = {'resnet': [backbone_feat_channels[-1], cfg]}
    model_kwargs = {'resnet': dict(norm_layer=norm_layer)}
    if encoder_type not in models:
        raise ValueError('Unknown encoder...')
    model = models[encoder_type.lower()]
    args = model_args[encoder_type.lower()]
    kwargs = model_kwargs[encoder_type.lower()]

    create_uni_en = False
    if cfg.MODEL.ENCODER.NUM_EN_FEATURES > 4:
        create_uni_en = True

    if create_uni_en:
        encoder = model(*args, **kwargs)
        uni_en_feat_channels = encoder.feat_channels
    else:
        encoder = None
        uni_en_feat_channels = []
    return encoder, uni_en_feat_channels


def get_decoder(decoder_type, norm_layer, en_feat_channels, cfg):
    expansion = DecoderBlock.expansion
    de_out_channels = cfg.MODEL.DECODER.OUTPLANES * expansion
    de_in_channels = [channels + de_out_channels for channels
                      in en_feat_channels[:-2]]
    de_in_channels += [en_feat_channels[-2:]]
    de_in_channels = list(reversed(de_in_channels))

    uni_decoder = Decoder(cfg, de_in_channels, norm_layer, decoder_type=decoder_type)
    return uni_decoder


def get_segment_head(cfg, num_en_features, name='seg_head', **kwargs):
    heads = {'seg_head': SegHead}
    head = heads.get(name, None)
    if head is not None:
        return head(cfg, cfg.NUM_CLASSES.SEGMENT, num_en_features)
    else:
        raise ValueError('Unknown head...')


def get_depth_head(cfg, num_en_features, name='depth_head', **kwargs):
    heads = {'depth_head': DepthHead}
    head = heads.get(name, None)
    if head is not None:
        return head(cfg, 1, num_en_features)
    else:
        raise ValueError('Unknown head...')


def get_sem_cont_head(cfg, num_en_features, **kwargs):
    if cfg.MISC.SEM_CONT_MULTICLASS:
        num_classes = cfg.NUM_CLASSES.SEGMENT
    else:
        num_classes = 1
    return SemanticContourHead(cfg, num_classes, num_en_features)


def get_sur_nor_head(cfg, num_en_features, **kwargs):
    return SurfaceNormalHead(cfg, num_en_features)


def get_autoencoder_head(cfg, num_en_features, **kwargs):
    return AutoEncoderHead(cfg, num_en_features, **kwargs)
