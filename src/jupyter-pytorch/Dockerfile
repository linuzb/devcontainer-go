# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG REGISTRY=quay.io
ARG OWNER=jupyter
ARG BASE_CONTAINER=$REGISTRY/$OWNER/scipy-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Installation de Code Server et server-proxy/vscode-proxy pour intégrer Code dans JupyterLab
# Install VSCode extensions
ENV CODE_VERSION=4.22.1
RUN curl -fOL https://github.com/coder/code-server/releases/download/v$CODE_VERSION/code-server_${CODE_VERSION}_amd64.deb && \
    dpkg -i code-server_${CODE_VERSION}_amd64.deb && \
    rm -f code-server_${CODE_VERSION}_amd64.deb && \
    EXT_LIST="ms-python.python ms-python.debugpy zhuangtongfa.material-theme" && \
    for EXT in $EXT_LIST; do code-server --install-extension $EXT; done

# Install PyTorch with pip (https://pytorch.org/get-started/locally/)
# hadolint ignore=DL3013
RUN pip install --no-cache-dir --index-url 'https://download.pytorch.org/whl/cu116' \
    'torch==1.12.0+cu116' \
    'torchvision==0.13.0+cu116 ' \
    'torchaudio==0.12.0+cu116'  && \
    pip install --no-cache-dir \
    d2l==0.17.6 && \
    conda install -y \
    jupyter-server-proxy \
    jupyter-vscode-proxy && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

USER ${NB_UID}

# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#dockerfiles
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility