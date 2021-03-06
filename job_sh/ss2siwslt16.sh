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

module load cuda/9.0.176
module load cudnn/7.3

source ~/allennlp/venv/bin/activate

allennlp train -f ~/entasum/simple_seq2seq_iwslt16.json --serialization-dir /gs/hs0/tga-nlp-titech/matsumaru/entasum/model/ss2s_iwslt16D400H400
