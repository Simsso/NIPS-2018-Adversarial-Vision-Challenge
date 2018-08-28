FROM tensorflow/tensorflow:latest

WORKDIR /opt/app

# required for cv2 (python-opencv)
RUN apt update && apt install -y libsm6 libxext6 && apt-get install -y libgtk2.0-dev

# Get some tf-gpu dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        libcudnn7=7.1.4.18-1+cuda9.0 \
        libnccl2=2.2.13-1+cuda9.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

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
