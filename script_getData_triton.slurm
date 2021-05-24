#!/bin/bash

sbatch -J text -o text.out script_getData_triton.sh text

sbatch -J call -o call.out script_getData_triton.sh call
