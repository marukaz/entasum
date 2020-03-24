#!/bin/sh
##n current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N bert
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.bert_loss
#$ -e e.bert_loss

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load intel nccl/2.4.2 cuda/10.0.130 cudnn

source /gs/hs0/tga-nlp-titech/matsumaru/repos/japanese-bert/venv/bin/activate

cd ~/home/repos/japanese-bert

python src/run_classifier.py   --task_name=anli   --do_train=true   --do_eval=true   --data_dir=/gs/hs0/tga-nlp-titech/matsumaru/entasum/data/ANLI_sep_rand    --model_file=model/wiki-ja.model   --vocab_file=model/wiki-ja.vocab   --init_checkpoint=model/model.ckpt-1400000   --max_seq_length=512   --train_batch_size=8   --num_train_epochs=3   --output_dir=model/ANLI_sep_rand_epoch3 --save_checkpoints_steps=2000
