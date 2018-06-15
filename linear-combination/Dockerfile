FROM tensorflow/tensorflow:latest

# download dataset
RUN mkdir /tmp/data && \
    cd /tmp/data && \
    curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz && \
    curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz && \
    curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz && \
    curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

WORKDIR /opt/app

COPY requirements.txt /opt/app

RUN pip install -r requirements.txt

COPY . /opt/app

EXPOSE 6006:6006

CMD ["python", "./src/main.py"]