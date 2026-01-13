# Use the official vllm image for gpu with Ampere architecture and above (Compute Capability>=8.0)
# Compute Capability version query (https://developer.nvidia.com/cuda-gpus)
FROM vllm/vllm-openai:v0.10.1.1

# Use the official vllm image for gpu with Turing architecture and below (Compute Capability<8.0)
# FROM vllm/vllm-openai:v0.10.2

# Install libgl for opencv support & Noto fonts for Chinese characters
RUN apt-get update && \
    apt-get install -y \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install mineru latest
RUN python3 -m pip install -U 'mineru[core]' --break-system-packages && \
    python3 -m pip cache purge

# Install some other packages
RUN python3 -m pip install -U \
        hydra-core \
        asciinet \
        tenacity \
        trafilatura \
        readability-lxml \
        html2text

# Install java
RUN wget https://mirrors.huaweicloud.com/openjdk/25/openjdk-25_linux-x64_bin.tar.gz -O /tmp/jdk-25.tar.gz && \
    mkdir -p /opt/jdk-25 && \
    tar -xzf /tmp/jdk-25.tar.gz -C /opt/jdk-25 --strip-components=1 && \
    rm /tmp/jdk-25.tar.gz

ENV JAVA_HOME=/opt/jdk-25
ENV PATH=$JAVA_HOME/bin:$PATH

# for deepsurvey
RUN pip install hydra-core && \
    pip install rich && \
    pip install loguru && \
    pip install tenacity && \
    pip install -U sentence-transformers && \
    pip install hdbscan && \

ENV HF_ENDPOINT=https://hf-mirror.com
# Download models and update the configuration file
RUN /bin/bash -c "mineru-models-download -s huggingface -m all"

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]