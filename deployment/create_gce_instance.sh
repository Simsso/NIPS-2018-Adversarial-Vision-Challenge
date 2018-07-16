#!/bin/bash

TERRAFORM = "terraform_0.11.7_darwin_amd64"

echo "Downloading Terraform .."
curl -O "https://releases.hashicorp.com/terraform/0.11.7/$TERRAFORM.zip"

echo "Installing Terraform .."
unzip "$TERRAFORM.zip"
mv terraform /usr/local/bin
chmod +x /usr/local/bin/terraform
rm "$TERRAFORM.zip"

echo "Initializing Terraform .."
terraform init