#!/bin/bash

sbatch -J alpha_99.0_t_2.0_k_100 -o alpha_99.0_t_2.0_k_100.out script_runModel.sh 99.0 2.0

sbatch -J alpha_99.0_t_10.0_k_100 -o alpha_99.0_t_10.0_k_100.out script_runModel.sh 99.0 10.0

sbatch -J alpha_99.0_t_100.0_k_100 -o alpha_99.0_t_100.0_k_100.out script_runModel.sh 99.0 100.0

sbatch -J alpha_99.0_t_1000.0_k_100 -o alpha_99.0_t_1000.0_k_100.out script_runModel.sh 99.0 1000.0

sbatch -J alpha_999.0_t_2.0_k_100 -o alpha_999.0_t_2.0_k_100.out script_runModel.sh 999.0 2.0

sbatch -J alpha_999.0_t_10.0_k_100 -o alpha_999.0_t_10.0_k_100.out script_runModel.sh 999.0 10.0

sbatch -J alpha_999.0_t_100.0_k_100 -o alpha_999.0_t_100.0_k_100.out script_runModel.sh 999.0 100.0

sbatch -J alpha_999.0_t_1000.0_k_100 -o alpha_999.0_t_1000.0_k_100.out script_runModel.sh 999.0 1000.0
