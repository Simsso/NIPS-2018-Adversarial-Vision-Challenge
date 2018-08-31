"""
Tiny ImageNet: Input Pipeline
Written by Patrick Coady (pcoady@alum.mit.edu)

Reads in jpegs, distorts images (flips, translations, hue and
saturation) and builds QueueRunners to keep the GPU well-fed. Uses
specific directory and file naming structure from data download
link below.

Also builds dictionary between label integer and human-readable
class names.

Get data here:
https://tiny-imagenet.herokuapp.com/
"""

import glob
import re
import tensorflow as tf
import random
import numpy as np
import os

DATASET_DIVISOR = 1
NUM_CLASSES = 200
NUM_TRAIN_SAMPLES = 500*NUM_CLASSES // DATASET_DIVISOR
NUM_VALIDATION_SAMPLES = 50*NUM_CLASSES // DATASET_DIVISOR
IMG_DIM = 64  # after cropping
IMG_CHANNELS = 3
PATH = os.path.expanduser('~/.data/tiny-imagenet-200')


def load_filenames_labels(mode):
    """Gets filenames and labels

    Args:
        mode: 'train' or 'val'
            (Directory structure and file naming different for
            train and val datasets)

    Returns:
        list of tuples: (jpeg filename with path, label)
  """
    label_dict, class_description = build_label_dicts()
    filenames_labels = []
    if mode == 'train':
        filenames = glob.glob(PATH + '/train/*/images/*.JPEG')
        for filename in filenames:
            match = re.search(r'n\d+', filename)
            label = str(label_dict[match.group()])
            filenames_labels.append((filename, label))
    elif mode == 'val':
        with open(PATH + '/val/val_annotations.txt', 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                filename = PATH + '/val/images/' + split_line[0]
                label = str(label_dict[split_line[1]])
                filenames_labels.append((filename, label))

    return filenames_labels


def build_label_dicts():
    """Build look-up dictionaries for class label, and class description

    Class labels are 0 to 199 in the same order as
    tiny-imagenet-200/wnids.txt. Class text descriptions are from
    tiny-imagenet-200/words.txt

    Returns:
        tuple of dicts
        label_dict:
            keys = synset (e.g. "n01944390")
            values = class integer {0 .. 199}
        class_desc:
            keys = class integer {0 .. 199}
            values = text description from words.txt
  """
    label_dict, class_description = {}, {}
    with open(PATH + '/wnids.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            synset = line[:-1]  # remove \n
            label_dict[synset] = i
    with open(PATH + '/words.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            synset, desc = line.split('\t')
            desc = desc[:-1]  # remove \n
            if synset in label_dict:
                class_description[label_dict[synset]] = desc

    return label_dict, class_description


def read_image(filename_q, mode):
    """Load next jpeg file from filename / label queue
    Randomly applies distortions if mode == 'train' (including a
    random crop to [56, 56, 3]). Standardizes all images.

    Args:
        filename_q: Queue with 2 columns: filename string and label string.
            filename string is relative path to jpeg file. label string is text-
            formatted integer between '0' and '199'
            mode: 'train' or 'val'

    Returns:
        [img, label]:
            img = tf.uint8 tensor [height, width, channels]  (see tf.image.decode.jpeg())
            label = tf.unit8 target class label: {0 .. 199}
    """
    item = filename_q.dequeue()
    filename = item[0]
    label = item[1]
    file = tf.read_file(filename)
    img = tf.image.decode_jpeg(file, IMG_CHANNELS)

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
    img = tf.cast(img, tf.float32) - tf.constant(_CHANNEL_MEANS)

    label = tf.string_to_number(label, tf.int32)
    label = tf.cast(label, tf.uint8)

    return [img, label]


def batch_q(mode, batch_size):
    """Return batch of images using filename Queue

    Args:
        mode: 'train' or 'val'
        batch_size: number of samples per batch

    Returns:
        imgs: tf.uint8 tensor [batch_size, height, width, channels]
        labels: tf.uint8 tensor [batch_size,]

    """
    filenames_labels = load_filenames_labels(mode)
    random.shuffle(filenames_labels)
    filename_q = tf.train.input_producer(filenames_labels, shuffle=True)

    # 2 read_image threads to keep batch_join queue full:
    result = tf.train.batch_join([read_image(filename_q, mode) for _ in range(2)],
                                 batch_size, shapes=[(IMG_DIM, IMG_DIM, IMG_CHANNELS), ()], capacity=2048)
    return result
