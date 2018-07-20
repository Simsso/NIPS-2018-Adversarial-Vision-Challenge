#!/bin/bash

echo "Install Ansible .."
apt-get update -y
apt-add-repository ppa:ansible/ansible
apt-get install -y ansible

echo "Configure Google Compute Engine Instance at $1 .."
ansible-playbook gce_playbook.yml  --extra-vars "host=$1"
