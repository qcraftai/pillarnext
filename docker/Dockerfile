FROM nvidia/cuda:11.6.1-devel-ubuntu20.04
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install basics
RUN apt update
RUN apt install -y wget git ffmpeg libsm6 libxext6 

ENV CONDA_DIR /opt/conda

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh && \
    /bin/bash ./Anaconda3-2021.11-Linux-x86_64.sh -b -p /opt/conda && \
    rm ./Anaconda3-2021.11-Linux-x86_64.sh 

ENV PATH="/opt/conda/bin:$PATH"
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
Run pip install hydra-core numba 
Run pip install spconv-cu116
Run pip install opencv-python
Run pip install nuscenes-devkit
Run pip install protobuf==3.20.*
RUN pip install --no-cache-dir tensorflow==2.6.0
Run pip install waymo-open-dataset-tf-2-6-0