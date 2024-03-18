ARG BASE_IMATE=nvidia/cuda:11.6.1-devel-ubuntu20.04

FROM  ${BASE_IMATE}

LABEL pytorch_version=torch==1.12.0+cu116,torchvision==0.13.0+cu116,torchaudio==0.12.0

LABEL ubuntu_software=build-essential,wget,pandoc,vim \
      conda_package=conda-pack,jupyter,jupyterlab


# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install base utilities
RUN apt-get update --yes \
    && apt-get install --yes --no-install-recommends \
    build-essential \
    wget \
    curl \
    # - pandoc is used to convert notebooks to html files
    #   it's not present in the aarch64 Ubuntu image, so we install it here
    pandoc \
    vim \
    git \
    python3-pip \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip && \
    pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116 && \
    pip install d2l==0.17.6 jupyter

# install code server and install extensions
RUN curl -fsSL https://code-server.dev/install.sh | sh \
    && code-server --install-extension ms-python.python \
	&& code-server --install-extension ms-python.vscode-pylance \
    && code-server --install-extension ms-toolsai.jupyter \
    && apt-get clean && rm -rf /var/lib/apt/lists/*