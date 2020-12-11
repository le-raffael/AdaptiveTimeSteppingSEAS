#!/bin/bash
cd ..
rm build -rf
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/raffael/local -DLUA_INCLUDE_DIR=/usr/local/include ..
make
cd ..
cd scriptsResults


