FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

WORKDIR /app/tts

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PATH="/usr/local/cuda/bin:/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

RUN apt-get update && apt-get install -y software-properties-common git wget curl vim libsndfile1 sox libsox-fmt-mp3 && rm -rf /var/lib/apt/lists/*

RUN wget \
	https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
	&& mkdir /root/.conda \
	&& bash Miniconda3-latest-Linux-x86_64.sh -b \
	&& rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda install -c conda-forge python=3.9.16

COPY src/requirements.txt src/requirements.txt

RUN pip3 install -r src/requirements.txt

COPY . .

CMD ["python", "app.py"]
