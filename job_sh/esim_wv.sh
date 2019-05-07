#!/bin/sh
##n current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=12:00:00
#$ -N esim_wv
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.esim_wv
#$ -e e.esim_wv

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/allennlp/venv/bin/activate

allennlp train -f ~/entasum/train_config/esim_wv.jsonnet --serialization-dir /gs/hs0/tga-nlp-titech/matsumaru/entasum/model/esim_wv_all_jamul_test
