FROM tensorflow/tensorflow:latest-gpu

WORKDIR /opt/app

# load Tiny ImageNet
RUN mkdir ~/.data && \
    cd ~/.data && \
    curl -O http://cs231n.stanford.edu/tiny-imagenet-200.zip && \
    unzip -qq tiny-imagenet-200.zip 

COPY requirements.txt /opt/app
RUN pip install -r requirements.txt

COPY . /opt/app

EXPOSE 6006:6006

CMD ["python", "./src/main.py"]
