#!/bin/bash
#SBATCH -o CNS_calls.out

srun python script_getData.py

#to run on cluster:
#sbatch script_getData.sh
