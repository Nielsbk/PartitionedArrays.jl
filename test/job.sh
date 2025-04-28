#!/bin/bash
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --output=das6-job.o
#SBATCH --error=das6-job.e
#SBATCH -C gpunode

source ./modules.sh
export OMP_NUM_THREADS=1
export NUM_THREADS=1
MPIFLAGS="--map-by node:span --rank-by core"
JULIAFLAGS="-O3 --check-bounds=no"
mpiexec -np 4 $MPIFLAGS julia $JULIAFLAGS --project=. niels_test.jl