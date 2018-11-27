#!/bin/bash

# Make sure to do "chmod +x setup.sh"
# Run this with "sudo ./setup.sh"

sudo apt update
sudo apt install -y pkg-config
sudo apt install -y gcc
sudo apt install -y gtk+2.0
sudo apt install -y libtiff5-dev
sudo apt install -y ctags
sudo apt install -y libpangox-1.0

sudo apt install -y python3.6-dev
sudo apt install -y python-pip
sudo apt install -y python-sklearn
pip3 install tensorflow
# If installing tensorflow doesn't work, do "pip uninstall tensorflow" then
# "pip3 install tensorflow==1.5"
# This occurs on older CPUs
