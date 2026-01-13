FROM docker.v2.aispeech.com/sjtu/sjtu_chenlu-yangziyue-cuda_12.2.2-ubuntu_22.04-torch_2.6-cu_124_general:v1.0

COPY . ./app/auto-review-writing

WORKDIR /app

RUN cd auto-review-writing && \
    pip install --no-cache-dir -r requirements.txt && \
    cd .. && \
    rm -rf auto-review-writing


RUN pip install --no-cache-dir https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz


# if network fails, download locally and uncomment the following lines:
# COPY ./en_core_sci_lg-0.5.4.tar.gz /app/en_core_sci_lg-0.5.4.tar.gz 

# RUN pip install --no-cache-dir ./en_core_sci_lg-0.5.4.tar.gz