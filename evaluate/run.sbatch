#!/bin/bash
#SBATCH --job-name=hulu-puli-chat
#SBATCH --nodes=1
#SBATCH --nodelist=lolka
#SBATCH --ntasks-per-node=1                   # number of tasks per node
#SBATCH --gres=gpu:2                          # number of GPUs per node
#SBATCH --cpus-per-task=4                     # 4 CPU cores per GPU for dataloading
#SBATCH --mem=50G                             # memory per node
#SBATCH --partition=hi
#SBATCH --error=/home/osvathm/HuLU/logs/job.%x.%J.err
#SBATCH --output=/home/osvathm/HuLU/logs/job.%x.%J.out


export DATASETS=""
export SAVE_RESULT_PATH=""

export MODEL_PATH=""
echo "Model: $MODEL_PATH"

PYTHONPATH="." python3 src/cli.py --model-name $MODEL_PATH --tasks hucola --parameters ./parameters.json \
    --experiment-name llm-crit --run-name puli-chat-run-0 --eval-test true --save-results-path $SAVE_RESULT_PATH \
    --hp-search --hp-trials 20



# hucopa -> AutoModelForMultipleChoice error for decoder models