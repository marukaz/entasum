#!/bin/sh
##n current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N ss2sjiji
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o ss2sjiji.err
#$ -e ss2sjiji.out

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/allennlp/venv/bin/activate

allennlp train ~/entasum/simple_seq2seq.json --serialization-dir /gs/hs0/tga-nlp-titech/matsumaru/entasum/model/ss2sjiji
