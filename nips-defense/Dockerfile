FROM gcr.io/nips-2018-102018/nips-tensorflow-base-image:latest

ARG MODEL_ID_ARG
ARG BUCKET_NAME_ARG
ARG INSTANCE_NAME_ARG
ARG ZONE_ARG
ARG PROJECT_ID_ARG

# persist the args across images
ENV MODEL_ID ${MODEL_ID_ARG}
ENV BUCKET_NAME ${BUCKET_NAME_ARG}
ENV INSTANCE_NAME ${INSTANCE_NAME_ARG}
ENV PROJECT_ID ${PROJECT_ID_ARG}
ENV ZONE ${ZONE_ARG}

WORKDIR /opt/app

# copy python project
COPY . /opt/app

# create output folder for volume
RUN mkdir output

# mount bucket and start training
RUN chmod +x ./start.sh

ENTRYPOINT ["/bin/bash", "-c", "./start.sh"]