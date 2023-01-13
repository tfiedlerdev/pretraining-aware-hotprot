#!/bin/bash

apt update
apt-get install -y gcc-7 g++-7
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 0
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 0
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 1
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 1
update-alternatives --auto gcc
update-alternatives --auto g++
apt install software-properties-common --yes
apt update
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.debd
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y install cuda
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
pip install -r requirements.txt
python train.py 