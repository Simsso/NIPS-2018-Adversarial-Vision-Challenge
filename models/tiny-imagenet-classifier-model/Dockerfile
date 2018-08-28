FROM gcr.io/nips-2018-207619/nips-tensorflow-base-image:latest

WORKDIR /opt/app

# install python requirements
COPY requirements.txt /opt/app
RUN pip install -r requirements.txt

# copy python project
COPY . /opt/app

#expose port for tensorboard
EXPOSE 6006:6006

CMD ["python", "./src/main.py"]
