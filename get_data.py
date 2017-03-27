import numpy as np
import scipy.misc

import matplotlib.image
import just


def prep_images(images):
    if not isinstance(images, list):
        images = [images]
    X = np.array([scipy.misc.imresize(im, (72, 128, 3)) for im in images])
    X = np.moveaxis(X, -1, 1)
    return X


def get_training_xy(data_path="~/tracktrack/"):
    positions = list(just.iread(data_path + "positions.jsonl"))
    images = [matplotlib.image.imread(x) for x in just.glob(data_path + "im*.png")]
    m = min(len(images), len(positions))

    X = prep_images(images[-m:])
    positions = positions[-m:]
    y = np.array(positions)

    return X, y
