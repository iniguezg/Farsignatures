#!/bin/bash

module load anaconda

#analysis 2
# sbatch -J text -o text.out script_getData_triton.sh text
# sbatch -J call -o call.out script_getData_triton.sh call

#analysis 3
#sbatch -J params_data_parallel -o params_data_parallel.out script_getData_triton.sh

#analysis 5/9/12
sbatch script_getData_triton.sh
