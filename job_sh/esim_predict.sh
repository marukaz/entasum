#!/bin/sh
##n current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N esim_pred
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.esim_pred
#$ -e e.esim_pred

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/allennlp/venv/bin/activate

cd ~/entasum

JSON_PATH="/gs/hs0/tga-nlp-titech/matsumaru/data/jnc_fairseq_3snt_100k_test/valid.jsonl";
ESIM_OUT="100k_test_valid_esim_probs.out";
allennlp predict ~/home/entasum/model/esim_dbs31_12000/model.tar.gz $JSON_PATH --include-package esim_predictor --predictor esim --output-file ~/home/entasum/out/$ESIM_OUT --silent
