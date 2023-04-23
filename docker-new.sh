#!/bin/bash

name=fact
port=8000 # pass-thuru port (for port forwarding)
work=`pwd`

run()
{
    case "$1" in
    build)
        rm -rf docker/${name}
        mkdir -p docker/${name}
        #cp -r dataset docker/${name}/
        cp -r models docker/${name}/
        cp -r info4/log docker/${name}/
        cp -r info4/dataset docker/${name}/
        cp -r info4/images docker/${name}/
        cp -r info4/output docker/${name}/
        cp -r lib docker/${name}/
        cp -r options docker/${name}/
        cp -r scripts docker/${name}/
        cp -r *.py docker/${name}/
        cp -r *.txt docker/${name}/
        cp -r *.sh docker/${name}/
        sudo docker build -t pushdown99/${name} docker
        ;;
    push)
        sudo docker push pushdown99/${name}
        ;;
    run)
        sudo docker run -it --rm --runtime=nvidia pushdown99/${name} bash
        ;;
    *)
        echo ""
        echo "Usage: docker-build {build|push|torch}"
        echo ""
        echo "       build : build docker image to local repositories"
        echo "       push  : push to remote repositories (docker hub)"
        echo "       run   : running docker image"
        echo ""
        return 1
        ;;
    esac
}
run "$@"

