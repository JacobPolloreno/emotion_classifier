import cv2 as cv
import numpy as np
import os
import pandas as pd

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from typing import Tuple, Generator


# FER2013
#   48 x 48 grayscale images of faces
_BOTTLENECK_FEATURES_DIR = 'data/bottleneck/'
_DATASET_DIR = 'data/'
_DATASET_PATH = _DATASET_DIR + 'fer2013.csv'
_IMAGE_DIM = (48, 48)
_DATA_SIZE = 35887


def preprocess_input(x: np.ndarray, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def get_batches(x: np.ndarray, y: np.ndarray, batch_size: int=32):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]


def get_data(path: str=_DATASET_PATH,
             image_dim: Tuple=_IMAGE_DIM,
             color: bool=False) -> Tuple[np.ndarray, np.ndarray]:

    # Load data
    data = pd.read_csv(path, usecols=['emotion', 'pixels'])
    data_size = len(data.pixels)
    if color:
        features = np.zeros(
            shape=(data_size, *image_dim, 3),
            dtype=np.float32)
    else:
        features = np.zeros(
            shape=(data_size, *image_dim),
            dtype=np.float32)
    labels = np.asarray(data.emotion, dtype=np.int32)

    for idx, row in enumerate(data.pixels):
        image_array = np.fromstring(
            str(row),
            dtype=np.uint8,
            sep=' ').reshape((48, 48))

        # Convert to color
        if color:
            image = cv.cvtColor(image_array, cv.COLOR_GRAY2RGB)
        else:
            image = image_array

        # Resize if needed
        if image_dim != (48, 48):
            image = cv.resize(image, image_dim)

        features[idx] = image

    # VGG Preprocess
    if color:
        features = preprocess_input_vgg(features)
    else:
        features = preprocess_input(features)

    return features, labels


def get_data_batches(path: str=_DATASET_PATH,
                     image_dim: Tuple=_IMAGE_DIM,
                     batch_size: int=32,
                     color: bool=False) -> Generator:

    # Load data
    data = pd.read_csv(path, usecols=['emotion', 'pixels'])
    data_size = len(data.pixels)

    # Load label data
    labels = np.asarray(data.emotion, dtype=np.int32)

    # Create a features array of (batch_size, *image_dim, channels)
    for ii in range(0, data_size, batch_size):
        # We want all data not just full batches
        if data_size - ii < batch_size:
            batch_size = data_size - ii

        if color:
            features = np.empty(
                shape=(batch_size, *image_dim, 3),
                dtype=np.float32)
        else:
            features = np.empty(
                shape=(batch_size, *image_dim),
                dtype=np.float32)

        for i, row in enumerate(data.pixels[ii:ii + batch_size]):
            image_array = np.fromstring(
                str(row),
                dtype=np.uint8,
                sep=' ').reshape((48, 48))

            # Convert to color
            if color:
                image = cv.cvtColor(image_array, cv.COLOR_GRAY2RGB)
            else:
                image = image_array

            # Resize if needed
            if image_dim != (48, 48):
                image = cv.resize(image, image_dim)

            features[i] = image

        # VGG Preprocess
        if color:
            features = preprocess_input_vgg(features)
        else:
            features = preprocess_input(features)

        yield features, labels[ii:ii + batch_size]


def create_bottleneck_feats(save_dir: str=_BOTTLENECK_FEATURES_DIR,
                            batch_size: int=32):
    # Load VGG16 model, remove top
    model = VGG16(weights='imagenet',
                  include_top=False,
                  pooling='avg')

    features = np.empty(
        shape=(_DATA_SIZE, 512),
        dtype=np.float32)
    labels = np.empty(
        shape=(_DATA_SIZE,),
        dtype=np.int32)

    batches = get_data_batches(
        color=True,
        image_dim=(224, 224),
        batch_size=batch_size)
    start = end = 0

    # TODO(Add progress bar)
    for x, y in batches:
        end = start + x.shape[0]
        features[start:end] = model.predict(x)
        labels[start:end] = y
        start = end

    # Create save dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Save to dir
    np.savez(save_dir + 'bottleneck_features_vgg16.npz',
             features=features, labels=labels)

    print(f"Bottleneck files created in \'{save_dir}\'.")
