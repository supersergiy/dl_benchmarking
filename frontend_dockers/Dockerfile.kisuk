from nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update; apt-get install -y git python-pip 
RUN apt-get install -y libboost-all-dev python-tk vim
WORKDIR /opt

RUN git clone https://github.com/torms3/DataTools \
 && cd DataTools \
 && pip install -r requirements.txt \
 && make

RUN git clone -b refactoring https://github.com/torms3/DataProvider \
 && cd DataProvider/python \
 && pip install -r requirements.txt \
 && make
RUN ls
RUN git clone https://github.com/supersergiy/Superhuman
RUN pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl  \
    && pip install torchvision 
