#!/bin/bash

if [ ! -d "./build" ]; then
    mkdir build
fi

cd build
cmake ..

if [ $? -ne 0 ]; then
    echo "There are some errors in process cmake."
    exit 1
fi

make -j32

if [ $? -ne 0 ]; then
    echo "There are some errors in process make."
    exit 1
fi

make install

if [ $? -ne 0 ]; then
    echo "There are some errors in process make install."
    exit 1
fi

echo "running a test."
cd ../executable
./01_programming_model/01_hello_world

if [ $? -ne 0 ]; then
    echo "There are some errors in process test."
    exit 1
else
    echo "Install Successfully."
fi