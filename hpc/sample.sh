#!/bin/sh
#BSUB -q gpuv100
#BSUB -J sample
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/sample.out
#BSUB -e hpc/logs/sample.err

# Sample sequences from a trained model:   bsub < hpc/sample.sh

source hpc/env.sh

uv run python sample.py --model mlp

