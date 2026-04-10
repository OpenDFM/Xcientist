FROM docker.v2.aispeech.com/sjtu/sjtu_chenlu-yangziyue-cuda_12.2.2-ubuntu_22.04-torch_2.6-cu_124_surveyforge:v0.3

WORKDIR /app

# 1. 复制整个Repo
COPY ./requirements.txt .

# 2. 安装依赖并随后删除Repo
RUN  pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install llama-index-embeddings-huggingface && \
    rm -rf /app/requirements.txt