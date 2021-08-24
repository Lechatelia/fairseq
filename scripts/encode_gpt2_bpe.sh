#!/bin/bash

# DATAPATH='/nfs/zhujinguo/datasets/data/bert_pretrain_data/wikitext/'
DATAPATH='/mnt/lustre/share_data/zhujinguo/data/bert_pretrain_data/wikitext'

# mkdir -p gpt2_bpe
# wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
# wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe

for SPLIT in train valid test; do \
    python examples/roberta/multiprocessing_bpe_encoder.py \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs $DATAPATH/wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs $DATAPATH/wikitext-103-raw/wiki.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done