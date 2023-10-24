#!/bin/bash

cd build
cmake ..

make -j32
make install

cd ../executable
./01_programming_model/01_hello_world