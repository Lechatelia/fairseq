#!/bin/bash

if [[ $# -ge 1 ]]; then
  JOB_NAME=${1}
else
  JOB_NAME=debug
fi

filename=$(basename "$0")
echo "$filename"
WORK_DIR=${filename//.sh//$JOB_NAME}
WORK_DIR=workdirs/$WORK_DIR
mkdir  -p $WORK_DIR


now=$(date +"%Y%m%d_%H%M%S")

TASKS=('MNLI' 'QNLI' 'QQP'  'RTE'  'SST-2' 'MRPC' 'CoLA' 'STS-B')
NUM_CLASSES=('3' '2' '2' '2' '2' '2' '2' '1')
LRS=('1e-5' '1e-5' '1e-5' '2e-5' '1e-5' '1e-5' '1e-5' '2e-5')
BATCHSIZES=('32' '32' '32' '16' '32' '16' '16' '16')
TOTAL_NUM_UPDATES=('123873'	'33112'	'113272'	'2036'	'20935'	'2296'	'5336'	'3598' )
WARMUP_UPDATES=('7432'	'1986'	'28318'	'122'	'1256'	'137'	'320'	'214')


MAX_SENTENCES=16        # Batch size.
ROBERTA_PATH=pretrained_checkpoints/roberta.large/model.pt
DATA_DIR=/nfs/zhujinguo/datasets/data/bert_pretrain_data/glue_data



for task_index in "${!TASKS[@]}"
 do 
  TASK=${TASKS[$task_index]}
  mkdir  -p $WORK_DIR/$TASK

  if [ "$TASK" = "STS-B" ]
     then
      checkpointmetric='--regression-target --best-checkpoint-metric loss'
     else
      checkpointmetric='--best-checkpoint-metric accuracy --maximize-best-checkpoint-metric'
  fi

    echo $TASK ${NUM_CLASSES[$task_index]} ${LRS[$task_index]}  ${BATCHSIZES[$task_index]} \
     ${TOTAL_NUM_UPDATES[$task_index]} ${WARMUP_UPDATES[$task_index]} $checkpointmetric


  CUDA_VISIBLE_DEVICES=0 python ./train.py ${DATA_DIR}/${TASK}-bin/ \
      --restore-file $ROBERTA_PATH \
      --max-positions 512 \
      --batch-size $MAX_SENTENCES \
      --max-tokens 4400 \
      --task sentence_prediction \
      --reset-optimizer --reset-dataloader --reset-meters \
      --required-batch-size-multiple 1 \
      --init-token 0 --separator-token 2 \
      --arch roberta_large \
      --criterion sentence_prediction \
      --num-classes ${NUM_CLASSES[$task_index]} \
      --dropout 0.1 --attention-dropout 0.1 \
      --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
      --clip-norm 0.0 \
      --lr-scheduler polynomial_decay --lr ${LRS[$task_index]} \
       --total-num-update ${TOTAL_NUM_UPDATES[$task_index]} \
      --warmup-updates ${WARMUP_UPDATES[$task_index]} \
      --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
      --max-epoch 10 --find-unused-parameters --save-dir $WORK_DIR/$TASK/checkpoints \
      $checkpointmetric \
    2>&1 | tee -a $WORK_DIR/$TASK/exp_$now.txt 
    
 done