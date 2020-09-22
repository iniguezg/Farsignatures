#!/bin/bash
#SBATCH --job-name=CNS_calls
#SBATCH --output=/scratch/cs/networks/%u/prg/xocial/Farsignatures/CNS_calls.%j.out
#SBATCH -p debug
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=200

module load anaconda

srun script_getData.py

#to run on Triton:
#sbatch script_getData.sh
