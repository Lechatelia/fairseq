#!/bin/bash

# DATAPATH='/nfs/zhujinguo/datasets/data/bert_pretrain_data/bookswiki/'
DATAPATH='/mnt/lustre/share_data/zhujinguo/data/bert_pretrain_data/bookswiki'

# mkdir -p gpt2_bpe
# wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
# wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe

srun -p cpu \
python examples/roberta/multiprocessing_bpe_encoder.py \
    --encoder-json gpt2_bpe/encoder.json \
    --vocab-bpe gpt2_bpe/vocab.bpe \
    --inputs $DATAPATH/bookswiki.doc \
    --outputs $DATAPATH/bookswiki.bpe \
    --keep-empty \
    --workers 60; \

srun -p cpu \
python examples/roberta/multiprocessing_bpe_encoder.py \
    --encoder-json gpt2_bpe/encoder.json \
    --vocab-bpe gpt2_bpe/vocab.bpe \
    --inputs $DATAPATH/bookswiki-1000.doc \
    --outputs $DATAPATH/bookswiki-1000.bpe \
    --keep-empty \
    --workers 60; \