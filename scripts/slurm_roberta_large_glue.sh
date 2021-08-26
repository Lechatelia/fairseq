#!/bin/bash


JOB_NAME=${1:-debug}

TASK_ARGS=${2:-"ALL"}
SRUN=${3:-"srun"}
PY_ARGS=${@:4}

GPUS=1
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


now=$(date +"%Y%m%d_%H%M%S")

TASKS=('MNLI' 'QNLI' 'QQP'  'RTE'  'SST-2' 'MRPC' 'CoLA' 'STS-B')
NUM_CLASSES=('3' '2' '2' '2' '2' '2' '2' '1')
LRS=('1e-5' '1e-5' '1e-5' '2e-5' '1e-5' '1e-5' '1e-5' '2e-5')
BATCHSIZES=('32' '32' '32' '16' '32' '16' '16' '16')
TOTAL_NUM_UPDATES=('123873'	'33112'	'113272'	'2036'	'20935'	'2296'	'5336'	'3598' )
WARMUP_UPDATES=('7432'	'1986'	'28318'	'122'	'1256'	'137'	'320'	'214')


MAX_SENTENCES=16        # Batch size.
ROBERTA_PATH=pretrained_checkpoints/roberta.large/model.pt
DATA_DIR=/mnt/lustre/share_data/zhujinguo/data/bert_pretrain_data/glue_data


for task_index in "${!TASKS[@]}"
 do 
  TASK=${TASKS[$task_index]}
  mkdir  -p $WORK_DIR/$TASK
  # 如果指定任务 就只评测指定的task
  if [ "$TASK_ARGS" != "ALL" -a "$TASK" != "$TASK_ARGS" ]; then
      echo skip  $TASK only run "$TASK_ARGS"
      continue
  fi 
  if [ "$TASK" = "STS-B" ]
     then
      checkpointmetric='--regression-target --best-checkpoint-metric loss'
     else
      checkpointmetric='--best-checkpoint-metric accuracy --maximize-best-checkpoint-metric'
  fi

    echo $TASK ${NUM_CLASSES[$task_index]} ${LRS[$task_index]}  ${BATCHSIZES[$task_index]} \
     ${TOTAL_NUM_UPDATES[$task_index]} ${WARMUP_UPDATES[$task_index]} $checkpointmetric

    while true # find unused tcp port
    do
        PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
        status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
        if [ "${status}" != "0" ]; then
            break;
        fi
    done
    # export MASTER_PORT=${PORT}

set -x 

a=$(echo $HOSTNAME | cut  -c12-16)
if [ $a == '198-6' -o "${SRUN}" == "spring" ]; then
    spring.submit arun --mpi=None  --job-name=${TASK}-${JOB_NAME} -n$GPUS --gpu   \
    --gres=gpu:${GPUS_PER_NODE}  --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task $CPUS_PER_TASK \
    " python ./train.py ${DATA_DIR}/${TASK}-bin/ \
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
      --weight-decay 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
      --clip-norm 0.0 \
      --lr-scheduler polynomial_decay --lr ${LRS[$task_index]} \
      --total-num-update ${TOTAL_NUM_UPDATES[$task_index]} \
      --warmup-updates ${WARMUP_UPDATES[$task_index]} \
      --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
      --max-epoch 10 --find-unused-parameters --save-dir $WORK_DIR/$TASK/checkpoints \
      $checkpointmetric  --distributed-port ${PORT} --distributed-world-size $GPUS $PY_ARGS \
      2>&1 | tee -a $WORK_DIR/$TASK/exp_$now.txt "
elif [ $a == '198-8' ]; then
    srun --partition=vc_research_2 --mpi=pmi2 \
  --job-name=${TASK}-${JOB_NAME} -n$GPUS \
  --gres=gpu:${GPUS_PER_NODE}  --ntasks-per-node=${GPUS_PER_NODE} \
  --kill-on-bad-exit=1  --cpus-per-task $CPUS_PER_TASK \
   python ./train.py ${DATA_DIR}/${TASK}-bin/ \
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
      --weight-decay 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
      --clip-norm 0.0 \
      --lr-scheduler polynomial_decay --lr ${LRS[$task_index]} \
       --total-num-update ${TOTAL_NUM_UPDATES[$task_index]} \
      --warmup-updates ${WARMUP_UPDATES[$task_index]} \
      --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
      --max-epoch 10 --find-unused-parameters --save-dir $WORK_DIR/$TASK/checkpoints \
      $checkpointmetric  \
      --distributed-port ${PORT} --distributed-world-size $GPUS $PY_ARGS \
    2>&1 | tee -a $WORK_DIR/$TASK/exp_$now.txt 
else
  echo only SH1986 and SH1988 supported now 
fi


sleep 10
    
 done