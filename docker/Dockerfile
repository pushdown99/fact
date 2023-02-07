FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV CUDA_VERSION 11.1.1

LABEL maintainer "URBAN UNION <haeyun@gmail.com>"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y apt-utils 2>&1 | \
    grep -v "^debconf: delaying package configuration, since apt-utils.*"

RUN apt install build-essential -y --no-install-recommends

RUN apt-get install -yq python3 python3-pip vim wget git

RUN apt-get install -yq libgl1 libglib2.0-0

RUN apt-get update

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

RUN python -m pip install --upgrade pip
RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --root-user-action=ignore pyyaml timer ipython matplotlib climage opencv-python==4.6.0.66 easydict sympy click Cython h5py tqdm ipykernel nltk gensim

RUN python -m ipykernel install --user --name torch --display-name "torch"

ADD fact /fact

COPY motd /etc/motd
COPY getpubip /bin
COPY jupyter-run /root
COPY bashrc /root/.bashrc
COPY vimrc /root/.vimrc

RUN echo 'clear' >> /etc/bash.bashrc
RUN echo '[ ! -z "$TERM" -a -r /etc/motd ] && cat /etc/motd' >> /etc/bash.bashrc
RUN echo 'ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime' >> /etc/bash.bashrc

RUN apt-get install -yq graphviz graphviz-dev
RUN pip install gensim GPUtil tabulate graphviz
RUN pip install --root-user-action=ignore numpy==1.23.1 --upgrade

WORKDIR /fact
RUN ln -sf nia/images images

