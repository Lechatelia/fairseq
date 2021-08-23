#!/bin/bash

DATA_DIR=/nfs/zhujinguo/datasets/data/bert_pretrain_data/wikitext/wikitext-103-raw/data-bin/wikitext-103

fairseq-hydra-train -m --config-dir examples/roberta/config/pretraining \
--config-name base task.data=$DATA_DIR
#  optimization.update_freq=[32]