#!/bin/bash

#NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=demo      #Set the job name 
#SBATCH --account=132778103316
#SBATCH --time=00:30:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks-per-node=1              #Request 1 task
#SBATCH --nodes=1
#SBATCH --mem=8gb                  #Request 2560MB (2.5GB) per node
#SBATCH --output=output/%x.%j      #Send stdout/err to "output/log.%j"
#SBATCH --gres=gpu:1                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=zhishguo@tamu.edu    #Send all emails to email_address

#First Executable Line
# module load CUDA/11.3.1
module load Anaconda3/2021.11
source ~/.bashrc
conda activate mytorch 

python3 demo.py --data=breastmnist --task_index=0 --pos_class=0 --epochs=100
sleep 5 

## --data=breastmnist --data=pneumoniamnist
