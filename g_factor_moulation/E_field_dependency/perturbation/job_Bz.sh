#!/bin/bash
#SBATCH --job-name=gpert_Bz
#SBATCH --nodelist=x9               # run on node x9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           # = Pool workers hard-coded in gfactor_Bz.py
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
##SBATCH --partition=<partition>    # uncomment and set if your cluster needs it

# B || z : perturbative (pymablock) + numerical (exact diagonalization) g_zz vs Ey and Ez
# Outputs: gz_moulation_E{y,z}.dat and gfactor_zz_vs_E{y,z}_N{N}_L300.dat
# Usage:   sbatch job_Bz.sh

set -euo pipefail

SCRIPT=gfactor_Bz.py

# --- environment ---
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

# parallelism = multiprocessing Pool; keep BLAS single-threaded per worker
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"   # needs Hamiltonian_mathematica_v2.py and numerical_gfactor.py here

echo "Job $SLURM_JOB_ID on $(hostname): $SCRIPT, $(date)"
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" python "$SCRIPT"
echo "Finished $(date)"
