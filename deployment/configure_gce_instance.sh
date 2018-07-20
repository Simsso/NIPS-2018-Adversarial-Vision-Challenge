#!/bin/bash

echo "Install Ansible .."
apt-get update -y
apt-add-repository ppa:ansible/ansible
apt-get install -y ansible

echo "Configure GCE Instance SSH Keys .."
cp nips-cloudbuilding $HOME/.ssh/nips-cloudbuilding
ssh-add $HOME/.ssh/nips-cloudbuilding

echo "Generate Ansible Host Inventory .."
printf "[nips_training]\n $1\n" > gcp-hosts

echo "Disable Ansible Host Key Checking .."
export ANSIBLE_HOST_KEY_CHECKING=false

echo "Configure Google Compute Engine Instance at $1 .."
ansible-playbook -i gcp-hosts gce_playbook.yml