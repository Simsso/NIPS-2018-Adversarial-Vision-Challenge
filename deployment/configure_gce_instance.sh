#!/bin/bash

echo "Install Ansible .."
sudo apt-add-repository ppa:ansible/ansible
sudo apt-get update
sudo apt-get install ansible

echo "Configure Google Compute Engine Instance at $1 .."
ansible-playbook gce_playbook.yml  --extra-vars "host=$1"
