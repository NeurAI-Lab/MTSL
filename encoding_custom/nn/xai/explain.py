import logging

from configs.defaults import _C as cfg
from encoding_custom.nn.xai import *
from encoding_custom.nn.xai import xai_utils
from utilities.infer_utils import setup_infer_model


known_explainers = {'CKA': CKA}


class ExplainerWrapper:
    def __init__(self, model, tasks, dl_val, device):
        explainer_args = getattr(cfg, 'EXPLAINER_ARGS', {})
        self.explainer = explainer(args, cfg, model, tasks, dl_val, device,
                                   **explainer_args)

    def run(self):
        self.explainer.explain()


def main():
    logging.getLogger().setLevel(logging.INFO)
    model, tasks, dl_val, device = setup_infer_model(
        args, cfg, num_workers=args.workers, batch_size=args.batch_size,
        collate_fn='train_collate', freeze_cfg=False)
    attack_wrap = ExplainerWrapper(model, tasks, dl_val, device)
    attack_wrap.run()


if __name__ == "__main__":
    parser = xai_utils.parse_args(known_explainers.values())
    args = parser.parse_args()
    args.split = 'val'
    explainer = known_explainers.get(args.explainer_name, None)
    if explainer is None:
        raise ValueError('Unknown explainer...')
    main()
