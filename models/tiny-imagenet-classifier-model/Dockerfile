FROM gcr.io/nips-2018-207619/nips-tensorflow-base-image:latest

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

# mount bucket and start training
RUN chmod +x ./start.sh $BUCKET_NAME

CMD "./start.sh"
