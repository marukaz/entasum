#!/bin/sh
##n current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N deco_t
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.deco_t
#$ -e e.deco_t

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/allennlp/venv/bin/activate

allennlp train -f ~/entasum/train_config/decomposable_attention_t.jsonnet --serialization-dir /gs/hs0/tga-nlp-titech/matsumaru/entasum/model/deco_dbs31_1-25000_part1_trainable
