import os


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
