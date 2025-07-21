#!/bin/bash -l

#$ -P npbssmic        # Specify the SCC project name you want to use
#$ -l h_rt=11:00:00   # Specify the hard time limit for the job
#$ -N yolo-training   # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -l gpus=1          # Num GPUs 
#$ -l gpu_c=8.0       # Compute capability
#$ -pe omp 16         # num CPU cores

conda activate detection_env
cd /projectnb/npbssmic/ac25/Defect-Detection
python -m src.training.tune