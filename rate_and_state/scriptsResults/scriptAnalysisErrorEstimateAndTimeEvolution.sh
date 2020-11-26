#!/bin/bash
cd ..
rm build -rf
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/raffael/local ..
make
echo "execute RKF45:"
./rs PID RKF45
echo "execute BDF12:"
./rs PID BDF12
echo "execute BDF23:"
./rs PID BDF23
cd ..
cd scriptsResults


