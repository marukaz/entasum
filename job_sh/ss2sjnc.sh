#!/bin/sh
##n current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N ss2sjnc
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o out.ss2sjnc
#$ -e err.ss2sjnc

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/allennlp/venv/bin/activate

allennlp train -f ~/entasum/train_config/simple_seq2seq_jnc.json --serialization-dir /gs/hs0/tga-nlp-titech/matsumaru/entasum/model/ss2sjnc
