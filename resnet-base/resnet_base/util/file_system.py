import os


def create_dir_if_not_present(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
