#!/bin/bash
sudo apt-get update
sudo apt-get install build-essential -y
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/390.25/NVIDIA-Linux-x86_64-390.25.run
sudo bash NVIDIA-Linux-x86_64-390.25.run
