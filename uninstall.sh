#!/bin/bash

if [ ! -d "./build" ];then
    echo "Not found build files."
else
    rm -rf ./build
fi

if [ ! -d "./executable" ];then
    echo "Not found executable files."
else
    rm -rf ./executable
fi

echo "Uninstall Successfully."