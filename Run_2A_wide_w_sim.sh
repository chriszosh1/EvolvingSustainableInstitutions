#!/bin/bash
#SBATCH --job-name=wide_bundleV2A_w_similarity
#SBATCH --output=wide_bundleV2A_w_similarity_output.txt
#SBATCH --error=wide_bundleV2A_w_similarity_error.txt
#SBATCH -N 1
#SBATCH -c 8

if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  omp_threads=$SLURM_CPUS_PER_TASK
else
  omp_threads=1
fi

export OMP_NUM_THREADS=$omp_threads

python3 wide_bundleV2A_w_similarity.py