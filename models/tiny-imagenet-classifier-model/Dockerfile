FROM gcr.io/nips-2018-207619/nips-tensorflow-base-image:latest

ARG LOG_FOLDER_ARG
ARG BUCKET_NAME_ARG

# persist the args across images
ENV LOG_FOLDER ${LOG_FOLDER_ARG}
ENV BUCKET_NAME ${BUCKET_NAME_ARG}

WORKDIR /opt/app

# install python requirements
COPY requirements.txt /opt/app
RUN pip install -r requirements.txt

# copy python project
COPY . /opt/app

# expose port for tensorboard
EXPOSE 6006:6006

# create folder for network mount
RUN mkdir -p ${BUCKET_NAME}
RUN mkdir -p ${BUCKET_NAME}/${LOG_FOLDER}

# mount bucket and start training
RUN chmod +x ./start.sh

ENTRYPOINT ["/bin/bash", "-c", "./start.sh ${BUCKET_NAME} ${LOG_FOLDER}"]
