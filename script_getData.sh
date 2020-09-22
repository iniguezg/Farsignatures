#!/bin/bash
#SBATCH -o CNS_calls.out

srun script_getData.py

#to run on cluster:
#sbatch script_getData.sh
