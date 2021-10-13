ARG TF_SERVING_BUILD_IMAGE=tensorflow/tensorflow:latest-gpu

FROM ${TF_SERVING_BUILD_IMAGE} 

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64"

ARG user

RUN useradd -ms /bin/bash $user \
        && apt-get update && apt-get install -y \
        libgl1-mesa-dev \
        && apt-get install -y git python3-pip python3.6 \
        && apt-get install -y openslide-tools \
        && apt-get install -y python3-openslide \
        && apt-get clean \
        && pip install --upgrade pip Pillow opencv-python openslide-python

USER $user

WORKDIR /home/$user

RUN git clone https://github.com/AlexZhurkevich/Drug-prediction-with-Electromagnetic-Data

WORKDIR /home/$user/Drug-prediction-with-Electromagnetic-Data

LABEL maintainer="Alex_Zhurkevich"
