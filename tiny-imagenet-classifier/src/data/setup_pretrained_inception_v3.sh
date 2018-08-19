#!/bin/sh
TF_RECORD_DATASET_DIR=~/.data/tiny-imagenet-tfrecords
DATASET_DIRECTORY=~/.data/tiny-imagenet-200 # not to be edited
CHECKPOINT_DIRECTORY=~/.models # not to be edited

# download the dataset if not already done
if [ ! -d "$DATASET_DIRECTORY" ]; then
    mkdir ~/.data 
    cd ~/.data
    echo "Tiny ImageNet Download started..."
    curl -sS http://cs231n.stanford.edu/tiny-imagenet-200.zip > tiny-imagenet-200.zip
    echo "Unzipping..."
    unzip tiny-imagenet-200.zip
    rm tiny-imagenet-200.zip
else
    echo "The Tiny ImageNet data set has already been downloaded."
    echo "Remove $DATASET_DIRECTORY and re-run this script to download it again. (rm -r $DATASET_DIRECTORY)"
fi

# convert it to tf-records format
if [ ! -d "$TF_RECORD_DATASET_DIR" ]; then
    echo "Converting Tiny ImageNet to TFRecords format..."
    python3 convert_tiny_imagenet_tfrecords.py --output_directory=$TF_RECORD_DATASET_DIR
    echo "Done converting."
else
    echo "TFRecords format already existent. Remove $TF_RECORD_DATASET_DIR to convert it again."
fi

# download the checkpoint file
if [ ! -d "$CHECKPOINT_DIRECTORY" ]; then
    mkdir ~/.models
    cd ~/.models
    echo "Downloading Inception v3 checkpoint..."
    curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
    tar xzf inception-v3-2016-03-01.tar.gz
    rm inception-v3-2016-03-01.tar.gz
    echo "Done downloading checkpoint."
else
    echo "Inception v3 checkpoint files already existent. Remove $CHECKPOINT_DIRECTORY to download again."
fi
