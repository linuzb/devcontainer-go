ARG BASE_CONTAINER=cubefs/cfs-server:v3.3.2

FROM $BASE_CONTAINER

RUN apt-get update --yes \
    && apt-get install --yes --no-install-recommends \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
