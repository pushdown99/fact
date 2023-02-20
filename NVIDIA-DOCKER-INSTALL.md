## Docker:

1. setting up docker

~~~bash

curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker

~~~

2. setting up nvidia container toolkit (repositories and GPG key setup)

~~~bash

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

~~~

3. package install

~~~bash

$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2
$ sudo systemctl restart docker
$ sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

~~~
