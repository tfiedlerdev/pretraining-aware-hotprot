FROM matthewfeickert/docker-python3-ubuntu

RUN sudo apt update
RUN sudo apt-get install -y gcc-7 g++-7
RUN sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 0
RUN sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 0
RUN sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 1
RUN sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 1
RUN sudo update-alternatives --auto gcc
RUN sudo update-alternatives --auto g++
RUN sudo apt install software-properties-common --yes
RUN sudo apt update
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
RUN sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
RUN sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
RUN sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
RUN sudo apt-get update
RUN sudo DEBIAN_FRONTEND=noninteractive apt-get -y install cuda
RUN git clone https://github.com/LeonHermann322/hot-prot.git
RUN pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
RUN pip install -r hot-prot/requirements.txt
