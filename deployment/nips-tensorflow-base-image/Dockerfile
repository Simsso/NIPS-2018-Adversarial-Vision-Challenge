FROM tensorflow/tensorflow:latest-gpu

# add CUDA path
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64/

# load Tiny ImageNet
RUN mkdir ~/.data && \
    cd ~/.data && \
    curl -O http://cs231n.stanford.edu/tiny-imagenet-200.zip && \
    unzip -qq tiny-imagenet-200.zip

# install gcsfuse
ENV GCSFUSE_REPO=gcsfuse-xenial

RUN echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# get service account for gcs
COPY cloudbuild-service-account.json /opt/app/cloudbuild-service-account.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/opt/app/cloudbuild-service-account.json

# install gcsfuse
RUN apt-get -y update 
RUN apt-get -y install gcsfuse