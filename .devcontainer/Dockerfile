FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

ARG USERNAME=vscode
ARG ALIRE_VERSION=2.1.0

# Install uv (pinned version)
COPY --from=ghcr.io/astral-sh/uv:0.9.27 /uv /uvx /usr/local/bin/

# Set uv environment variables
ENV UV_LINK_MODE=copy

# Install some common tools and dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends fzf libgmp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Alire (Ada/SPARK package manager)
RUN ARCH=$(uname -m) \
    && curl -Lo /tmp/alr.zip \
    "https://github.com/alire-project/alire/releases/download/v${ALIRE_VERSION}/alr-${ALIRE_VERSION}-bin-${ARCH}-linux.zip" \
    && unzip /tmp/alr.zip -d /tmp/alr \
    && mv /tmp/alr/bin/alr /usr/local/bin/alr \
    && chmod +x /usr/local/bin/alr \
    && rm -rf /tmp/alr /tmp/alr.zip

# Install Ada/SPARK tools via Alire
USER $USERNAME
ENV TMPDIR=/tmp
ENV PATH="/home/$USERNAME/.alire/bin:${PATH}"
RUN alr install gnat_native=15.2.1 gprbuild=25.0.1 gnatprove=15.1.0 \
    && alr -C ~ get gnatformat=26.0.0 \
    && alr -n -C ~/gnatformat_26.0.0_a0a869fa build -- -j12 \
    && mv ~/gnatformat_26.0.0_a0a869fa/bin/gnatformat ~/.alire/bin/gnatformat \
    && rm -rf ~/gnatformat_26.0.0_a0a869fa

# Install Python 3.13 via uv
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
RUN uv python install 3.13

# Install some coding agents
RUN curl -fsSL https://claude.ai/install.sh | bash \
    && curl -fsSL https://opencode.ai/install | bash

# Bash history persistence
USER root
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory \
    && printf '\n# Bash history persistence\n' >> "/home/$USERNAME/.bashrc" \
    && echo "$SNIPPET" >> "/home/$USERNAME/.bashrc"

# Final user
USER $USERNAME
