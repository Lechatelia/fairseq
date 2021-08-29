#!/bin/bash


JOB_NAME=${1:-debug}

GPUS=${2:-8}
SRUN=${3:-"srun"}
PY_ARGS=${@:4}


GPUS_PER_NODE=${GPUS:-8}
if [ $GPUS_PER_NODE -ge 8 ]; then
  GPUS_PER_NODE=8
fi

CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}

filename=$(basename "$0")
echo "$filename"
WORK_DIR=${filename//.sh//$JOB_NAME}
WORK_DIR=workdirs/$WORK_DIR
mkdir  -p $WORK_DIR

a=$(echo $HOSTNAME | cut  -c12-16)
if [ $a == '198-6' ]; then
DATA_DIR=/mnt/lustre/share_data/zhujinguo/data/bert_pretrain_data/bookswiki/data-bin/bookswiki-combine
elif [ $a == '198-8' ]; then
DATA_DIR=/mnt/cache/share_data/zhujinguo/data/bert_pretrain_data/bookswiki/data-bin/bookswiki-combine
else
  echo only SH1986 and SH1988 supported now 
fi

now=$(date +"%Y%m%d_%H%M%S")

TOTAL_UPDATES=500000    # Total number of training steps
WARMUP_UPDATES=1000    # Warmup the learning rate over this many updates
PEAK_LR=0.00125          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=32        # Number of sequences per batch (batch size)
UPDATE_FREQ=2          # Increase the batch size 16x

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
# export MASTER_PORT=${PORT}

a=$(echo $HOSTNAME | cut  -c12-16)
if [ $a == '198-6' -o "${SRUN}" == "spring" ]; then
spring.submit arun --mpi=None  --job-name=${JOB_NAME} -n$GPUS --gpu   \
--gres=gpu:${GPUS_PER_NODE}  --ntasks-per-node=${GPUS_PER_NODE} \
--cpus-per-task $CPUS_PER_TASK \
" python ./train.py --fp16 $DATA_DIR --fp16-no-flatten-grads \
    --task masked_lm --criterion masked_lm \
    --arch perceiver_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer lamb --lamb-betas '(0.9,0.999)' --lamb-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler cosine --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES  \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 5 --save-dir $WORK_DIR/checkpoints \
    --distributed-port ${PORT} --distributed-world-size $GPUS $PY_ARGS \
    2>&1 | tee -a $WORK_DIR/exp_$now.txt "

elif [ $a == '198-8' ]; then
 srun --partition=vc_research_2 --mpi=pmi2 \
  --job-name=${JOB_NAME} -n$GPUS \
  --gres=gpu:${GPUS_PER_NODE}  --ntasks-per-node=${GPUS_PER_NODE} \
  --kill-on-bad-exit=1  --cpus-per-task $CPUS_PER_TASK \
   python ./train.py --fp16 $DATA_DIR --fp16-no-flatten-grads \
    --task masked_lm --criterion masked_lm \
    --arch perceiver_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer lamb --lamb-betas '(0.9,0.999)' --lamb-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler cosine --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES  \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 5 --save-dir $WORK_DIR/checkpoints \
    --distributed-port ${PORT} --distributed-world-size $GPUS $PY_ARGS \
    2>&1 | tee -a $WORK_DIR/exp_$now.txt 
else
  echo only SH1986 and SH1988 supported now 
fi