#!/bin/bash

echo "Install Ansible .."
apt-get update -y
apt-add-repository ppa:ansible/ansible
apt-get install -y ansible

echo "Generate Ansible Host Inventory .."
printf "[nips_training]\n $1\n" > gcp-hosts

echo "Disable Ansible Host Key Checking .."
export ANSIBLE_HOST_KEY_CHECKING=false

echo "Configure Instance at $1 and start model training .."
ansible-playbook -i gcp-hosts gce_playbook.yml --key-file nips-cloudbuilder --extra-vars "model_docker_image=$2"