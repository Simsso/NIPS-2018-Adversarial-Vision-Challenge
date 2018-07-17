#!/bin/bash

echo "Install Ansible .."
apt-add-repository ppa:ansible/ansible
apt-get update
apt-get install ansible

echo "Configure Google Compute Engine Instance at $1 .."
ansible-playbook gce_playbook.yml  --extra-vars "host=$1"
