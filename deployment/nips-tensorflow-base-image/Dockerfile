FROM tensorflow/tensorflow:latest-gpu-py3

# add CUDA path
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64/

# load Tiny ImageNet
RUN mkdir ~/.data && \
    cd ~/.data && \
    curl -O http://cs231n.stanford.edu/tiny-imagenet-200.zip && \
    unzip -qq tiny-imagenet-200.zip

# get service account for gcs
COPY cloudbuild-service-account.json /opt/app/cloudbuild-service-account.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/opt/app/cloudbuild-service-account.json