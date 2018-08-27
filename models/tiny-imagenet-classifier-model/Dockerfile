FROM tensorflow/tensorflow:latest

WORKDIR /opt/app

# required for cv2 (python-opencv)
RUN apt update && apt install -y libsm6 libxext6 && apt-get install -y libgtk2.0-dev

# load Tiny ImageNet
RUN mkdir ~/.data && \
    cd ~/.data && \
    curl -O http://cs231n.stanford.edu/tiny-imagenet-200.zip && \
    unzip tiny-imagenet-200.zip -qq

COPY requirements.txt /opt/app
RUN pip install -r requirements.txt

COPY . /opt/app

EXPOSE 6006:6006

CMD ["python", "./src/main.py"]
