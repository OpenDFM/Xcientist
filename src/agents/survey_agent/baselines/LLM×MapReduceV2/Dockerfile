FROM docker.v2.aispeech.com/sjtu/sjtu_chenlu-yangziyue-cuda_12.2.2-ubuntu_22.04-torch_2.6-cu_124_general:v1.0

COPY . ./app/LLMxMapReduce

WORKDIR /app

RUN cd LLMxMapReduce/LLMxMapReduce_V2 && \
    pip --no-cache-dir install -r requirements.txt && \
    python3 -m playwright install --with-deps chromium && \
    cd .. && \
    cd .. && \
    rm -rf LLMxMapReduce

# FROM docker.v2.aispeech.com/sjtu/sjtu_chenlu-yangziyue-cuda_12.2.2-ubuntu_22.04-torch_2.6-cu_124_llm_mapreduce:v0.1

RUN pip install playwright && \
    playwright install