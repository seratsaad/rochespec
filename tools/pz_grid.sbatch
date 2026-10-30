#!/bin/bash
#SBATCH --job-name=cbpz-grid
#SBATCH --account=<project>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --array=0-14
#SBATCH --output=logs/grid-%A_%a.out
export CBPZ=${CBPZ:-$PWD}   # working directory with pz_grid.py and the output grid
module load python/3.12 2>/dev/null
source $CBPZ/venv/bin/activate
export PZ_ROOT=$HOME/payne-zero PYTHONPATH=$HOME/payne-zero NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export NUMBA_CACHE_DIR=$CBPZ/.numba_cache
cd $CBPZ
python pz_grid.py $SLURM_ARRAY_TASK_ID 15 grid
