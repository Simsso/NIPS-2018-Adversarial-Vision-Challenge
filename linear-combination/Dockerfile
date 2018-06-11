FROM tensorflow/tensorflow:latest

WORKDIR /opt/app

COPY requirements.txt /opt/app

RUN pip install -r requirements.txt

COPY . /opt/app

EXPOSE 6006:6006

CMD ["python", "./src/main.py"]