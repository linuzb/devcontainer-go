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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Installation de Code Server et server-proxy/vscode-proxy pour intégrer Code dans JupyterLab
# Install VSCode extensions
ENV CODE_VERSION=4.22.1
RUN curl -fOL https://github.com/coder/code-server/releases/download/v$CODE_VERSION/code-server_${CODE_VERSION}_amd64.deb && \
    dpkg -i code-server_${CODE_VERSION}_amd64.deb && \
    rm -f code-server_${CODE_VERSION}_amd64.deb
    # EXT_LIST="ms-python.python zhuangtongfa.material-theme" && \
    # for EXT in $EXT_LIST; do code-server --install-extension $EXT; done

# Copy environment.yml (if found) to a temp locaition so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
# COPY environment.yml noop.txt /tmp/conda-tmp/
# RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
#     && rm -rf /tmp/conda-tmp

# Configure environment
ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# Pin the Python version here, or set it to "default"
ARG PYTHON_VERSION=3.11

# Download and install Micromamba, and initialize the Conda prefix.
#   <https://github.com/mamba-org/mamba#micromamba>
#   Similar projects using Micromamba:
#     - Micromamba-Docker: <https://github.com/mamba-org/micromamba-docker>
#     - repo2docker: <https://github.com/jupyterhub/repo2docker>
# Install Python, Mamba, and jupyter_core
# Cleanup temporary files and remove Micromamba
# Correct permissions
# Do all this in a single RUN command to avoid duplicating all of the
# files across image layers when the permissions change
WORKDIR /tmp
RUN set -x && \
    arch=$(uname -m) && \
    if [ "${arch}" = "x86_64" ]; then \
        # Should be simpler, see <https://github.com/mamba-org/mamba/issues/1437>
        arch="64"; \
    fi && \
    # https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#linux-and-macos
    wget --progress=dot:giga -O - \
        "https://micro.mamba.pm/api/micromamba/linux-${arch}/latest" | tar -xvj bin/micromamba && \
    PYTHON_SPECIFIER="python=${PYTHON_VERSION}" && \
    if [[ "${PYTHON_VERSION}" == "default" ]]; then PYTHON_SPECIFIER="python"; fi && \
    # Install the packages
    ./bin/micromamba install \
        --root-prefix="${CONDA_DIR}" \
        --prefix="${CONDA_DIR}" \
        --yes \
        "${PYTHON_SPECIFIER}" \
        'mamba' \
        'jupyter_core' && \
    rm -rf /tmp/bin/ && \
    # Pin major.minor version of python
    # https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#preventing-packages-from-updating-pinning
    mamba list --full-name 'python' | tail -1 | tr -s ' ' | cut -d ' ' -f 1,2 | sed 's/\.[^.]*$/.*/' >> "${CONDA_DIR}/conda-meta/pinned" && \
    mamba clean --all -f -y

# # mamba/conda instsall
# RUN mamba install --yes \
#     'jupyter-server-proxy' \
#     'jupyter-vscode-proxy' && \
#     mamba clean --all -f -y && \
#     fix-permissions "${CONDA_DIR}" && \
#     fix-permissions "/home/${NB_USER}"

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements.txt noop.txt /tmp/pip-tmp/
RUN if [ -f "/tmp/pip-tmp/requirements.txt" ]; then pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt; fi \
   && rm -rf /tmp/pip-tmp \
   && fix-permissions "${CONDA_DIR}" \
   && fix-permissions "/home/${NB_USER}"

# USER ${NB_UID}
