#!/bin/bash
#SBATCH --partition=long                      # Ask for unkillable job
#SBATCH --cpus-per-task=10                    # Ask for 2 CPUs
#SBATCH --gres=gpu:40gb:3                         # Ask for 1 GPU
#SBATCH --mem=64G                             # Ask for 10 GB of RAM
#SBATCH --time=24:00:00                        # The job will run for 3 hours
#SBATCH -o /network/scratch/j/jarrid.rector-brooks/logs/dreamfold-%j.out  # Write the log on tmp1

source /home/mila/j/jarrid.rector-brooks/scratch/venvs/recursion_gfn/bin/activate
cd /home/mila/j/jarrid.rector-brooks/repos/recursion_gfn/src/gflownet/tasks

python seh_frag_cp.py
