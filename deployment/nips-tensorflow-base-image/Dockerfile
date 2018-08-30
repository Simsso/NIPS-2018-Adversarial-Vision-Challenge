FROM tensorflow/tensorflow:latest-gpu

# add CUDA path
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64/

# load Tiny ImageNet
RUN mkdir ~/.data && \
    cd ~/.data && \
    curl -O http://cs231n.stanford.edu/tiny-imagenet-200.zip && \
    unzip -qq tiny-imagenet-200.zip 