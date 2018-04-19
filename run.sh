#!/bin/bash

set -x
module load intel-parallel-studio
make clean && make
export MKL_NUM_THREADS=18
export KMP_AFFINITY=compact,1,0,granularity=fine

numactl -N 0 ./dgemm
numactl -N 1 ./dgemm
