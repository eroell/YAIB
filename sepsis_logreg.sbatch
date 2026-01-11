#!/bin/sh


#SBATCH --job-name=yaib_aki
#SBATCH -o logs/%j_slurm_output_gpu.txt
#SBATCH -e logs/%j_slurm_error_gpu.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00

source $HOME/.bashrc
conda activate yaib_from_ymlII

DATASET_NAME="eicu" # miiv mimic_demo
TASK="sepsis"
DATA_DIR="/home/icb/eljas.roellin/ehrapy_workspace/physionet-credentialized/code/eljas_stuff/YAIB-cohorts/data/${TASK}/${DATASET_NAME}"

TRAIN_DATASET_NAME=$DATA_DIR
VAL_DATASET_NAME=$DATA_DIR
TEST_DATASET_NAME=$DATA_DIR


LOG_DIR="/home/icb/eljas.roellin/ehrapy_workspace/physionet-credentialized/code/eljas_stuff/YAIB/custom_logs"

MODEL="logreg"

LOG_SUFFIX="${TASK}_${DATASET_NAME}_${MODEL}_v1"
python custom_main.py --data-dir $DATA_DIR --train-dataset-name $TRAIN_DATASET_NAME --val-dataset-name $VAL_DATASET_NAME --test-dataset-name $TEST_DATASET_NAME --log-dir $LOG_DIR --log-suffix $LOG_SUFFIX --model $MODEL --use-wandb

