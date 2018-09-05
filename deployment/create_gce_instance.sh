#!/bin/bash

echo "Retrieving model version and model folder name .."

MODEL_VERSION=$( echo $2 | grep -o '[0-9.]*$')
MODEL_FOLDER=${2%-$MODEL_VERSION}

TERRAFORM="terraform_0.11.7_linux_amd64"

echo "Install required packages .."
apt-get update -y
apt-get install -y curl
apt-get install -y zip

echo "Downloading Terraform .."
curl -O "https://releases.hashicorp.com/terraform/0.11.7/$TERRAFORM.zip"

echo "Installing Terraform .."
unzip "$TERRAFORM.zip"
mv terraform /usr/local/bin
chmod +x /usr/local/bin/terraform
rm "$TERRAFORM.zip"

echo "Initializing Terraform .."
cd deployment
terraform init

echo "Planning Terraform .."
terraform plan

echo "Applying Terraform plan .."
terraform apply -auto-approve -var "instance_name=$2" -var "model_docker_image=$MODEL_FOLDER:$MODEL_VERSION"