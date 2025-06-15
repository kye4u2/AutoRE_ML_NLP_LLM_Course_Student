FROM python:3.10-slim AS builder

# Set a non-interactive frontend for APT to avoid interactive prompts during the build
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      protobuf-compiler \
      bash curl zip unzip sed \
      wget sudo \
      python3-pip \
      git-lfs \
      openjdk-17-jdk \
      ssh \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Create and switch to non-root user for enhanced security
RUN useradd --create-home vscode
RUN usermod -aG sudo vscode \
 && usermod -s /bin/bash vscode \
 && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode && chmod 0440 /etc/sudoers.d/vscode

# Download and install Ghidra
RUN mkdir -p /opt \
 && wget -q -O /opt/ghidra_10.4_PUBLIC.zip https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_10.4_build/ghidra_10.4_PUBLIC_20230928.zip \
 && unzip -q /opt/ghidra_10.4_PUBLIC.zip -d /opt \
 && rm /opt/ghidra_10.4_PUBLIC.zip

ENV GHIDRA_HOME=/opt/ghidra_10.4_PUBLIC \
    PATH="$GHIDRA_HOME:$PATH"

# Default command to run when starting the container (optional, customize as needed)
CMD ["/bin/bash"]
