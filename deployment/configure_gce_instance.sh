#!/bin/bash

echo "Install Ansible .."
apt-get update -y
apt-add-repository ppa:ansible/ansible
apt-get install -y ansible

echo "Generate Ansible Host Inventory .."
printf "[nips_training]\n $1\n" > gcp-hosts

echo "Configure Google Compute Engine Instance at $1 .."
ansible-playbook -i gcp-hosts gce_playbook.yml 