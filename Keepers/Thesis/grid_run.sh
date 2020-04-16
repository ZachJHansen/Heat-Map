#!/bin/bash
#$ -V
#$ -cwd
#$ -S /bin/bash
#$ -N LSTM_Grid_3
#$ -o $JOB_NAME.o$JOB_ID
#$ -e $JOB_NAME.e$JOB_ID
#$ -q omni
#$ -pe sm 36
#$ -P quanah
#$ -l h_rt=48:00:00
#$ -l h_vmem=5.3G

module load intel python3
source ~/miniconda3/etc/profile.d/conda.sh
export HDF5_USE_FILE_LOCKING='FALSE'
conda init bash
conda activate tensorflow_env
python grid_lstm.py 0.5 3
