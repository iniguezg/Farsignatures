#!/bin/bash
#SBATCH --time=01-00
#SBATCH --partition=batch

srun python script_getData.py $1
