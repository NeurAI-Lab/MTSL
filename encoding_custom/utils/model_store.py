"""Model store which provides pretrained models."""
import os

from utilities.files import download

__all__ = ["get_model_file", "purge"]


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def get_url(name):
    url = model_urls.get(name, None)
    if url is None:
        raise ValueError(
            "Pretrained model for {name} is not available.".format(name=name))
    return url


def get_model_file(name, root=os.path.join("~", ".encoding", "models")):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot
     be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    url = get_url(name)
    file_name = url.split('/')[-1]
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name)
    if os.path.exists(file_path):
        return file_path
    else:
        print("Model file is not found. Downloading.")

    if not os.path.exists(root):
        os.makedirs(root)

    download(url, path=file_path, overwrite=True)

    return file_path


def purge(root=os.path.join("~", ".encoding", "models")):
    r"""Purge all pretrained model files in local file store.

    Parameters
    ----------
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".pth"):
            os.remove(os.path.join(root, f))
