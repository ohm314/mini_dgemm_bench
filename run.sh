#!/bin/bash

set -x

module purge
module load intel-parallel-studio
make clean && make
export MKL_DYNAMIC=false
export OMP_NESTED=true
export OMP_PROC_BIND=spread,close
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=18

numactl ./dgemm
