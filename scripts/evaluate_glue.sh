#!/bin/bash

if [ $# -ge 2 ]; then
  TASKS=${1} # ALL STS-B CoLA
  checkpoints=${2} # workdirs/slurm_roberta_large_glue/finetune
  SRUN=${3:-'srun'}
else
echo sh scripts/evaluate_glue.sh ALL workdirs/slurm_roberta_large_glue/finetune
fi

now=$(date +"%Y%m%d_%H%M%S")
a=$(echo $HOSTNAME | cut  -c12-16)

a=$(echo $HOSTNAME | cut  -c12-16)
if [ $a == '198-6' ]; then
DATA_DIR=/mnt/lustre/share_data/zhujinguo/data/bert_pretrain_data/glue_data
elif [ $a == '198-8' ]; then
DATA_DIR=/mnt/cache/share_data/zhujinguo/data/bert_pretrain_data/glue_data
else
  echo only SH1986 and SH1988 supported now 
fi

if [ "${SRUN}" == "server" ]; then


python scripts/evaluate_glue.py \
--checkpoints ${checkpoints} --tasks $TASKS --data_dir $DATA_DIR \
2>&1 | tee -a $checkpoints/glue_eval_$now.txt 

elif [ $a == '198-6' -o "${SRUN}" == "spring" ]; then

spring.submit arun --mpi=None  --job-name=glue-evaluate -n1 --gpu   \
--gres=gpu:1 --ntasks-per-node=1 --cpus-per-task 4 \
"
python scripts/evaluate_glue.py \
--checkpoints ${checkpoints} --tasks $TASKS --data_dir $DATA_DIR  \
2>&1 | tee -a $checkpoints/glue_eval_$now.txt "

elif [ $a == '198-8' ]; then
srun --partition=vc_research_2 --mpi=pmi2 \
  --job-name=glue-evaluate -n1\
  --gres=gpu:1  --ntasks-per-node=1 \
  --kill-on-bad-exit=1  --cpus-per-task 4 \
python scripts/evaluate_glue.py \
--checkpoints ${checkpoints} --tasks $TASKS --data_dir $DATA_DIR  \
2>&1 | tee -a $checkpoints/glue_eval_$now.txt 
else
  echo only SH1986 and SH1988 supported now 
fi
