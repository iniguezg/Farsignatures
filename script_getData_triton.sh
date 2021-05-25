#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=short

#analysis 2
# srun python script_getData.py $1

#analysis 3
srun python script_getData.py
