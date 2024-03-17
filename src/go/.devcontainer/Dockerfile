ARG BASE_IMAGE="mcr.microsoft.com/devcontainers/go:1.21-bookworm"

FROM ${BASE_IMAGE}

LABEL software.debian="curl,vim,ping,telnet,nc,dnsutils,traceroute,net-tools"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    vim \
    iputils-ping \
    telnet \
    netcat-traditional \
    dnsutils \
    traceroute \
    net-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install code server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# extensions to code-server
RUN code-server --install-extension golang.go
