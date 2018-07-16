from data import Data
from config import config
import numpy as np
import uuid
import cv2
from random import shuffle


class TinyImageNet:

    def __init__(self):
        self.data = Data()

    def load_class_ids(self):
        wnids_blob = self.data.get_file(config['wnids_path'], config['wnids_filename'])
        wnids_file = wnids_blob.download_as_string()
        return wnids_file.split('\n')

    def train_path_for_class_id(self, class_id):
        return "%s/%s/images/" % (config['train_folder'], class_id)

    def load_train_file_names(self, class_ids, images_per_class=config['train_count_per_class']):
        total_images_count = len(class_ids) * images_per_class

        files = []
        labels = np.zeros(total_images_count, dtype=np.int32)

        for index in range(len(class_ids)):
            class_id = class_ids[index]
            path = self.train_path_for_class_id(class_id)

            files += self.data.list_files(path)
            # TODO Need clarification for this
            labels[(index * images_per_class):((index + 1) * images_per_class)] = index

        return files, labels

    def load_image(self, path, size=config['image_size']):
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

    def get_train_data(self, shuffle_images=True):
        class_ids = self.load_class_ids()
        files, labels = self.load_train_file_names(class_ids=class_ids)

        total_count = len(files)
        images = []

        for i in range(total_count):
            file_name = str(uuid.uuid4())
            files[i].download_to_filename(file_name)

            if not i % 10000:  # print state for each 10k-th iteration (there are 100k in total)
                print("Loading tiny image net training set: %dk of %dk" % (i / 1000, total_count / 1000))

            images.append(self.load_image(file_name))

        if shuffle_images:
            c = list(zip(images, labels))
            shuffle(c)
            images, labels = zip(*c)

        images = np.array(images)
        labels = np.array(labels)

        return images, labels
