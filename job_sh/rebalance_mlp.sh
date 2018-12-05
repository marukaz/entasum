#!/bin/sh
##n current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N ss2siw
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o out.ss2siw
#$ -e err.ss2siw

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

export PYTHONPATH=$PYTHONPATH:~/entasum

module load cuda/9.0.176
module load cudnn/7.3

source ~/allennlp/venv/bin/activate

python ~/entasum/create_dataset/generate_candidates/rebalance_dataset_mlp.py