#FROM nvidia/cuda:latest
FROM nvidia/cuda:11.0-base-ubuntu20.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 python3-pip python3-setuptools python3-dev \
    gcc g++ libpython3.8-dev build-essential libpcre3 libpcre3-dev apt-utils git

WORKDIR .

COPY requirements.txt /src/requirements.txt
RUN pip3 install --no-cache-dir -r /src/requirements.txt

## still copies too much
COPY telepath_worker /src/telepath-core/telepath_worker
COPY telepath_gym /src/telepath-core/telepath_gym
COPY deepgym /src/telepath-core/deepgym
COPY syscheck /src/telepath-core/syscheck

# companion repositories, installed from GitHub
RUN pip3 install --no-cache-dir git+https://github.com/pathway/stable-baselines3-telepath.git
RUN git clone --depth 1 https://github.com/pathway/fqf-iqn-qrdqn.pytorch.git /src/fqf && \
    cd /src/fqf && pip3 install --no-cache-dir .
# telepath code imports `fqf.fqf_iqn_qrdqn`, so /src (the parent of /src/fqf) must be importable
ENV PYTHONPATH=/src

COPY setup.py /src/telepath-core/setup.py
RUN cd /src/telepath-core && pip3 install --no-cache-dir .
#RUN cd /src/telepath-core/ && pip3 install --no-cache-dir .

WORKDIR /src/telepath-core/telepath_worker/

ENV PATH="/src/telepath-core/telepath_worker/:${PATH}"

# dumb
RUN mkdir -p /tmp/telepath-runs/cartpole/replay/ && \
    mkdir -p /tmp/telepath-runs/cartpole/agents/ && \
    mkdir -p /tmp/telepath-runs/cartpole/agent_perf/ && \
    mkdir -p /src/telepath/mlruns && \
    chmod a+wrx / && \
    mkdir -p /nonexistent && \
    chmod a+wrx /nonexistent && \
    mkdir -p /src/telepath/mlruns

# ARGS are env vars
ENV COMMAND=fulltest
ENV ARGS=
ENV CONFIG_FILE=config/docker_cartpole.yaml

EXPOSE 6006
EXPOSE 80

# default cmd
CMD python3 telepath.py ${COMMAND} --config=${CONFIG_FILE} ${ARGS}

# Dockerfile : put often changed items later

# EC2Role: AWSBatchServiceRole  arn:aws:iam::123456789012:instance-profile/ecsInstanceRole
# Cloudformation yaml easier for setup?

# AWSBatch:
# JobDefn
# Compute env
# JobQueues

# IAM Service role
# ECR repo
# S3 bucket
