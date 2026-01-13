FROM docker.v2.aispeech.com/sjtu/sjtu_chenlu-yangziyue-cuda_12.2.2-ubuntu_22.04-torch_2.6-cu_124_surveyforge:v0.3

COPY . /app/auto_research
WORKDIR /app/auto_research

RUN apt update && \
    apt-get install libpq-dev python3-dev -y

RUN pip install --no-cache-dir -r write_paper/requirements.txt && \
    pip install --no-cache-dir -r sync_paper/requirements.txt