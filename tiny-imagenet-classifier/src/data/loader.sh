#!/bin/sh

DIRECTORY=/tmp/data/tiny-imagenet-200

if [ ! -d "$DIRECTORY" ]; then
    mkdir /tmp/data
    cd /tmp/data
    echo "Download started..."
    curl -sS http://cs231n.stanford.edu/tiny-imagenet-200.zip > tiny-imagenet-200.zip
    echo "Unzipping..."
    unzip tiny-imagenet-200.zip
    rm tiny-imagenet-200.zip
else
    echo "The Tiny ImageNet data set has already been downloaded."
    echo "Remove $DIRECTORY and re-run this script to download it again. (rm -r $DIRECTORY)"
fi