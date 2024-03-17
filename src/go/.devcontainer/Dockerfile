ARG baseImage="mcr.microsoft.com/devcontainers/go:1.21-bookworm"
FROM baseImage

RUN curl -fsSL https://code-server.dev/install.sh | sh

RUN code-server --install-extension golang.go
