FROM tensorflow/tensorflow:latest-gpu

WORKDIR /opt/app

# load Tiny ImageNet
RUN mkdir ~/.data && \
    cd ~/.data && \
    curl -O http://cs231n.stanford.edu/tiny-imagenet-200.zip && \
    unzip -qq tiny-imagenet-200.zip 

# install python requirements
COPY requirements.txt /opt/app
RUN pip install -r requirements.txt

# add CUDA path
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64/

COPY . /opt/app

#expose port for tensorboard
EXPOSE 6006:6006

CMD ["python", "./src/main.py"]
