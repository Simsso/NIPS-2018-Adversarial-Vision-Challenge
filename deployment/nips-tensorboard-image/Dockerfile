# Author: Samed Güner
# Tensorboad Image which accesses centralised tf_logs on GCS
# Recommended Run command:
# docker run -d -p 6006:6006 --rm gcr.io/nips-2018-207619/nips-tensorboard-image:latest
FROM tensorflow/tensorflow:latest

ARG BUCKET_NAME_ARG
ENV BUCKET_NAME ${BUCKET_NAME_ARG}

# add CUDA path
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64/

# install gcsfuse
ENV GCSFUSE_REPO=gcsfuse-xenial

RUN echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# install gcsfuse
RUN apt-get -y update 
RUN apt-get -y install gcsfuse

WORKDIR /opt/app

# get service account for gcs
COPY cloudbuild-service-account.json ./cloudbuild-service-account.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/opt/app/cloudbuild-service-account.json

# install tensorflow deps
COPY . ./
RUN pip install -r requirements.txt

# expose port for tensorboard
EXPOSE 80:80

# create folder for network mount
RUN mkdir -p ${BUCKET_NAME}

# mount bucket and start training
RUN chmod +x ./start.sh

# start tensorboard
ENTRYPOINT ["/bin/bash", "-c", "./start.sh"]