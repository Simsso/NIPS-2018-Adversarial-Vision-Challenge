
#!/bin/bash

echo "Retrieving model version and model folder name .."

MODEL_VERSION=$( echo $2 | grep -o '[0-9.]*$')
MODEL_FOLDER=${2%-$MODEL_VERSION}

echo "Model Folder: $MODEL_FOLDER"
echo "Model Version: $MODEL_VERSION"

echo "Update repositories .."
apt-get update 

echo "Update required libs .."
apt-get install -y apt-transport-https ca-certificates curl software-properties-common curl

echo "Add GPG keys .."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

echo "Add official Docker repository .."
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

echo "Update repositories .."
apt-get update 

echo "Install docker .."
apt-get install -y docker-ce

echo "Building Docker Image .."
docker build -t gcr.io/$1/$MODEL_FOLDER:$MODEL_VERSION  --build-arg BUCKET_NAME $3 models/$MODEL_FOLDER

echo "Pushing Docker Image .."
docker push gcr.io/$1/$MODEL_FOLDER:$MODEL_VERSION