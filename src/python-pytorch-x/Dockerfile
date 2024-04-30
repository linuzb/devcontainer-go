# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG REGISTRY=quay.io
ARG OWNER=jupyter
ARG BASE_CONTAINER=$REGISTRY/$OWNER/scipy-notebook
FROM $BASE_CONTAINER

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --yes \
    && apt-get install --yes --no-install-recommends \
    # build-essential \
    wget \
    curl \
    # - pandoc is used to convert notebooks to html files
    #   it's not present in the aarch64 Ubuntu image, so we install it here
    pandoc \
    vim \
    git \
    zip \
    unzip \
    tmux \
    screen \
    python3.9 \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN which pip3 \
    && pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu116 \
    torch==1.13.1+cu116 \
    torchvision==0.14.1+cu116 \
    torchaudio==0.13.1
    
# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# COPY requirements.txt noop.txt /tmp/pip-tmp/
# RUN if [ -f "/tmp/pip-tmp/requirements.txt" ]; then pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt; fi \
#    && rm -rf /tmp/pip-tmp

# USER ${NB_UID}
