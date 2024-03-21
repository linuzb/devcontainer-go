ARG BASE_IMATE=quay.io/jupyter/pytorch-notebook:cuda11-pytorch-2.2.1

FROM ${BASE_IMATE}

RUN pip install --no-cache-dir \
    'd2l==0.17.6'  && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
