#!/bin/bash
#PBS -A IscrC_DASHE
#PBS -j oe
#PBS -l walltime=23:45:00
#PBS -l select=1:ncpus=1:ngpus=1

module load profile/advanced;
module load python/3.5.2;
module load cuda/7.5.18;
module load gnu/4.9.2;
module load blas/3.6.0--gnu--4.9.2;
module load lapack/3.6.0--gnu--4.9.2;

export LD_LIBRARY_PATH=$WORK/cudnn_5/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=$WORK/cudnn_5/cuda/include:$CPATH
export LIBRARY_PATH=$WORK/cudnn_5/cuda/lib64:$LIBRARY_PATH

source $WORK/VirtualEnvs/fall_detection_env/bin/activate

python main_experiment.py -cf experiment.conf



 
