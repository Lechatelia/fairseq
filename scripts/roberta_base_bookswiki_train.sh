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

DATA_DIR=/nfs/zhujinguo/datasets/data/bert_pretrain_data/bookswiki/data-bin/bookswiki-combine

now=$(date +"%Y%m%d_%H%M%S")

TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x


python ./train.py --fp16 $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --save-dir $WORK_DIR/checkpoints \
    2>&1 | tee -a $WORK_DIR/exp_$now.txt 
