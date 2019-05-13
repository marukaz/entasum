#!/bin/sh
##n current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N bert_pred
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.bert_pred
#$ -e e.bert_pred

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load intel nccl/2.4.2 cuda/10.0.130 cudnn

source /gs/hs0/tga-nlp-titech/matsumaru/repos/japanese-bert/venv/bin/activate

cd ~/home/repos/japanese-bert

python src/run_classifier.py   --task_name=anli   --do_predict=true   --data_dir=/gs/hs0/tga-nlp-titech/matsumaru/entasum/data/jnc_bert_predict    --model_file=model/wiki-ja.model   --vocab_file=model/wiki-ja.vocab  --init_checkpoint=model/anli_shuf_model/model.ckpt-28141 --max_seq_length=512   --predict_batch_size=16  --output_dir /gs/hs0/tga-nlp-titech/matsumaru/entasum/data/jnc_bert_predict
