#!/bin/bash
#SBATCH --job-name=gfac_2HGe
#SBATCH --array=0-2                 # 0 = Bx, 1 = By, 2 = Bz (one task per script)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7          # = number of Pool workers in the .py (each solves one E-field point)
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
##SBATCH --partition=<partition>    # uncomment and set if your cluster needs it

# Usage:
#   sbatch -w x5 job.sh                          # run on node x5 (N = 100 default)
#   sbatch -w x7 job.sh                          # or any other free node
#   sbatch job.sh                                # let Slurm pick any free node
#   GF_N=60 sbatch -w x5 job.sh                  # other grid size (exported to the job)
#   sbatch -w x5 --array=2 job.sh                # only B || z on x5
#   sinfo -N -o "%N %t %C %e"                    # check which nodes are idle / free CPUs & mem

set -euo pipefail

SCRIPTS=(gfactor_Bx_variation_Ef_signed.py gfactor_By_variation_Ef_signed.py gfactor_Bz_variation_Ef_signed.py)
SCRIPT=${SCRIPTS[$SLURM_ARRAY_TASK_ID]}

# --- environment ---
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

export GF_N=${GF_N:-100}
# parallelism = multiprocessing Pool with SLURM_CPUS_PER_TASK workers (set in the .py);
# keep BLAS single-threaded per worker to avoid oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"   # needs Hamiltonian_mathematica_v2.py here too

echo "Job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname): $SCRIPT, GF_N=$GF_N, $(date)"
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" python "$SCRIPT"   # Slurm >=22.05: srun doesn't inherit this
echo "Finished $(date)"