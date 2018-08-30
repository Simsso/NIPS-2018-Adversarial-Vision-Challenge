FROM tensorflow/tensorflow:latest

WORKDIR /opt/app

COPY requirements.txt /opt/app

# required for cv2 (python-opencv)
RUN apt update && apt install -y libsm6 libxext6 && apt-get install -y libgtk2.0-dev

RUN pip install -r requirements.txt

COPY . /opt/app

EXPOSE 6006:6006

COPY cloudbuild-service-account.json cloudbuild-service-account.json
ENV GOOGLE_APPLICATION_CREDENTIALS cloudbuild-service-account.json

CMD ["python", "./src/main.py"]