from .mtsl import get_mtsl_arch


def get_multitask_model(name, **kwargs):
    models = {'mtsl_arch': get_mtsl_arch}
    return models[name.lower()](**kwargs)
