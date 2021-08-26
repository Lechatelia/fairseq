#!/bin/bash

DATA_DIR=/mnt/lustre/share_data/zhujinguo/data/bert_pretrain_data/glue_data
# DATA_DIR=/nfs/zhujinguo/datasets/data/bert_pretrain_data/glue_data
mkdir -p $DATA_DIR

# python scripts/download_glue_data.py --data_dir ${DATA_DIR} --tasks all
python scripts/download_glue_data.py --data_dir ${DATA_DIR} --tasks MRPC