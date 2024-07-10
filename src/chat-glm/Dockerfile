# FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04
FROM ubuntu:22.04

WORKDIR /workspace

RUN apt-get update --yes \
    && apt-get install --yes --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements.txt noop.txt /tmp/pip-tmp/
RUN if [ -f "/tmp/pip-tmp/requirements.txt" ]; then pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt; fi \
   && rm -rf /tmp/pip-tmp

# download model
RUN huggingface-cli download --resume-download THUDM/chatglm3-6b && \
    huggingface-cli download --resume-download suno/bark-small && \
    huggingface-cli download --resume-download openai/whisper-base

COPY run.py /workspace

CMD ["python3", "/workspace/run.py"]
