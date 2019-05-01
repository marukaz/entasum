#!/bin/sh
##n current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N sr_mlp
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.r_mlp
#$ -e e.r_mlp

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

export PYTHONPATH=$PYTHONPATH:~/entasum

module load cuda/9.0.176
module load cudnn/7.3

source ~/allennlp/venv/bin/activate

python ~/entasum/create_dataset/generate_candidates/rebalance_dataset_mlp.py ~/home/entasum/fairseq_model/jnc_3snt_transformer_wmtset_d01_upfreq2_gen/beam63_from_test_DBS_prefix0_snt3_wmt_d01_gpu4_updatefreq2.out -p ~/home/entasum/fairseq_model/jnc_tgt_transformer_lm_200k_test_gen/jnc_fairseq_dbs63_tf_200k_test_lm_3ktest_sorted.out
