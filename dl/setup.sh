#!/bin/bash

# Make sure to do "chmod +x setup.sh"
# Run this with "sudo ./setup.sh"

sudo apt-get update
sudo apt-get install -y pkg-config
sudo apt-get install -y gcc
sudo apt-get install -y gtk+2.0
sudo apt-get install -y libtiff5-dev
sudo apt-get install -y ctags
sudo apt-get install -y libpangox-1.0

sudo apt-get install -y python-dev
sudo apt-get install -y python-pip
pip install tensorflow
# If that doesn't work, do "pip uninstall tensorflow" then
# "pip install tensorflow==1.5"
# This occurs on older CPUs
