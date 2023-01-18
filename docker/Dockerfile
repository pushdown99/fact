FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y apt-utils 2>&1 | \
    grep -v "^debconf: delaying package configuration, since apt-utils.*"
RUN apt-get install -yq build-essential && \
    apt-get install -yq python3  && \
    apt-get install -yq python3-pip && \
    apt-get install -yq vim && \
    apt-get install -yq wget && \
    apt-get install -yq git && \
    apt-get install -yq libgl1 && \
    apt-get install -yq tree && \
    apt-get install -yq nmap && \
    apt-get install -yq tzdata && \
    apt-get install -yq dnsutils && \
    apt-get install -yq pciutils && \
    apt-get install -yq libglib2.0-0 && \
    apt-get install -yq net-tools
RUN pip3 install tensorflow tensorflow_addons opencv-python jupyter matplotlib awscli pandas scipy GPUtil pyyaml sympy h5py cython scipy numpy pytest-shutil tqdm ninja
RUN pip3 install torch torchvision torchaudio
RUN pip3 install flickrapi easydict sklearn menpo imageio pydot nltk graphviz
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN apt-get install -y --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
RUN apt-get install -y python3.8-venv graphviz
RUN apt-get update
COPY motd /etc/motd
COPY getpubip /bin
COPY bashrc /root/.bashrc
COPY vimrc /root/.vimrc

RUN echo 'clear' >> /etc/bash.bashrc
RUN echo '[ ! -z "$TERM" -a -r /etc/motd ] && cat /etc/issue && cat /etc/motd' >> /etc/bash.bashrc
RUN echo 'ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime' >> /etc/bash.bashrc
RUN echo 'cd /workspaces' >> /etc/bash.bashrc
RUN echo 'cp /root/jupyter-run' /workspaces >> /etc/bash.bashrc
RUN echo 'nvidia-smi' >> /etc/bash.bashrc
ADD fact /fact
