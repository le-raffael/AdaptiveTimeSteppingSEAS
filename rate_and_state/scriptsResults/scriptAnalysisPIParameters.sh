#!/bin/bash
cd ..
rm build -rf
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/raffael/local ..
make
echo "execute RKF45:"
./rs PID RKF45 PIparams
echo "execute BDF12:"
./rs PID BDF12 PIparams
echo "execute BDF23:"
./rs PID BDF23 PIparams
cd ..
cd scriptsResults


