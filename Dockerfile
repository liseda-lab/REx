# official tensorflow image with gpu 
FROM tensorflow/tensorflow:1.15.0-gpu-py3

# public key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    wget \
    git \
    curl \
    screen \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#install uv 
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /REx

CMD ["bash"]
