#!/bin/bash

if [[ $# -ge 1 ]]; then
  JOB_NAME=${1}
else
  JOB_NAME=debug
fi

if [ $# -ge 2 ]; then
  GPUS=$2
else
  GPUS=16
fi

GPUS_PER_NODE=${GPUS:-8}
if [ $GPUS_PER_NODE -ge 8 ]; then
  GPUS_PER_NODE=8
fi

CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:4}

filename=$(basename "$0")
echo "$filename"
WORK_DIR=${filename//.sh//$JOB_NAME}
WORK_DIR=workdirs/$WORK_DIR
mkdir  -p $WORK_DIR

DATA_DIR=/mnt/lustre/share_data/zhujinguo/data/bert_pretrain_data/bookswiki/data-bin/bookswiki-combine

now=$(date +"%Y%m%d_%H%M%S")

TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=8        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

spring.submit arun --mpi=None  --job-name=${JOB_NAME} -n$GPUS --gpu   \
--gres=gpu:${GPUS_PER_NODE}  --ntasks-per-node=${GPUS_PER_NODE} \
--cpus-per-task $CPUS_PER_TASK \
" python ./train.py --fp16 $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --arch roberta_large --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas \"(0.9,0.98)\" --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --save-dir $WORK_DIR/checkpoints \
    --distributed-port ${PORT} $PY_ARGS \
    2>&1 | tee -a $WORK_DIR/exp_$now.txt "
