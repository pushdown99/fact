#!/bin/bash

name=fact
port=8000 # pass-thuru port (for port forwarding)

run()
{
    case "$1" in
    build)
        rm -rf docker/${name}
        mkdir -p docker/${name}
        cp -r dataset docker/${name}/
        cp -r models docker/${name}/
        cp -r lib docker/${name}/
        cp -r options docker/${name}/
        cp -r scripts docker/${name}/
        cp -r *.py docker/${name}/
        cp -r *.txt docker/${name}/
        cp -r *.sh docker/${name}/
        sudo docker build -t pushdown99/fact docker
        ;;
    push)
        sudo docker push pushdown99/fact
        ;;
    run)
        host="${name}-P${port}"
        sudo docker run -p ${port}:${port}/tcp --name ${host} -h ${host} --ipc=host --mount type=bind,source=/home/hyhwang/repositories/model/fact/dataset,target=/fact/dataset --mount type=bind,source=/home/hyhwang/repositories/dataset/NIA/download/origin,target=/fact/images --mount type=bind,source=/home/hyhwang/repositories/model/fact/output,target=/fact/output -it --rm --runtime=nvidia pushdown99/fact bash
        ;;
    *)
        echo ""
        echo "Usage: docker-build {build|push|torch}"
        echo ""
        echo "       build : build docker image to local repositories"
        echo "       push  : push to remote repositories (docker hub)"
        echo ""
        return 1
        ;;
    esac
}
run "$@"

