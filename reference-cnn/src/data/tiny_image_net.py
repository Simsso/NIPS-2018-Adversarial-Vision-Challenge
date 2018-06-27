import cv2
import numpy as np
import glob
from random import shuffle

DATASET_ROOT = "tiny_image_net"

WNIDS_PATH = DATASET_ROOT + "/wnids.txt"
TRAIN_FOLDER = DATASET_ROOT + "/train"

CLASS_COUNT = 200
TRAIN_COUNT_PER_CLASS = 500

IMAGE_SIZE = 64


def load_class_ids():
    with open(WNIDS_PATH, "r") as wnids_file:
        class_ids = wnids_file.readlines()
        class_ids = [x.strip() for x in class_ids]
    return class_ids


def train_path_for_class_id(class_id):
    return TRAIN_FOLDER + "/" + class_id + "/images/*.JPEG"


def load_train_file_names(class_ids, images_per_class=TRAIN_COUNT_PER_CLASS):
    total_images_count = CLASS_COUNT * images_per_class

    files = []
    labels = np.zeros(total_images_count, dtype=np.int)
    for index in range(len(class_ids)):
        # load image paths
        class_id = class_ids[index]
        path = train_path_for_class_id(class_id)

        class_files = glob.glob(path)
        assert len(class_files) == images_per_class

        files = files + class_files
        labels[(index * images_per_class):((index + 1)*images_per_class)] = index

    return files, labels


def load_image(path, size=IMAGE_SIZE):
    img = cv2.imread(path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


"""
Returns the tuple of images and labels in the Tiny ImageNet Training Set.
Takes about 1-2 minutes to load the images into RAM.

Result:
- images: shape (100000, 64, 64, 3)  type: np.float32
- labels: shape (100000,)            type: np.int
"""
def get_train_data(shuffle_images=True):
    class_ids = load_class_ids()
    files, labels = load_train_file_names(class_ids=class_ids)

    total_count = len(files)
    images = []
    for i in range(total_count):
        if not i % 10000:   # print state for each 10k-th iteration (there are 100k in total)
            print("Loading tiny image net training set: %dk of %dk" % (i / 1000, total_count / 1000))
        images.append(load_image(files[i]))

    if shuffle_images:
        c = list(zip(images, labels))
        shuffle(c)
        images, labels = zip(*c)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

