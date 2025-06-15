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
      apt-transport-https \
      ssh \
 && rm -rf /var/lib/apt/lists/*

#Install Adoptium to bring down jdk 21
RUN wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor | tee /etc/apt/trusted.gpg.d/adoptium.gpg > /dev/null
RUN echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list
RUN apt-get update && apt-get install -y temurin-21-jdk

WORKDIR /workspace

# Create and switch to non-root user for enhanced security
RUN useradd --create-home vscode
RUN usermod -aG sudo vscode \
 && usermod -s /bin/bash vscode \
 && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode && chmod 0440 /etc/sudoers.d/vscode

# Download and install Ghidra 11.2.1
RUN mkdir -p /opt \
 && wget -q -O /opt/ghidra_11.2.1_PUBLIC_20241105.zip https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.2.1_build/ghidra_11.2.1_PUBLIC_20241105.zip \
 && unzip -q /opt/ghidra_11.2.1_PUBLIC_20241105.zip -d /opt \
 && rm /opt/ghidra_11.2.1_PUBLIC_20241105.zip

ENV GHIDRA_HOME=/opt/ghidra_11.2.1_PUBLIC_20241105 \
    PATH="$GHIDRA_HOME:$PATH"

RUN wget -q -O /opt/ghidra_11.2.1_PUBLIC_20250111_Blackfyre.zip /opt/https://github.com/jonescyber-ai/Blackfyre/releases/download/v1.0.1/ghidra_11.2.1_PUBLIC_20250111_Blackfyre.zip

# Default command to run when starting the container (optional, customize as needed)
CMD ["/bin/bash"]
