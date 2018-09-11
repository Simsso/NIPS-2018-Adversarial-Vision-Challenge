FROM gcr.io/nips-2018-207619/nips-tensorflow-base-image:latest

ARG LOG_FOLDER
ARG BUCKET_NAME

WORKDIR /opt/app

# install python requirements
COPY requirements.txt /opt/app
RUN pip install -r requirements.txt

# copy python project
COPY . /opt/app

# expose port for tensorboard
EXPOSE 6006:6006

# create folder for network mount
RUN mkdir $BUCKET_NAME
RUN mkdir $BUCKET_NAME/$LOG_FOLDER

# mount bucket and start training
RUN chmod +x ./start.sh

CMD ./start.sh $BUCKET_NAME/$LOG_FOLDER
