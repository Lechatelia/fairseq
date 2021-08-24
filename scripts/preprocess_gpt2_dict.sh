#!/bin/bash
# DATAPATH='/nfs/zhujinguo/datasets/data/bert_pretrain_data/wikitext/'
DATAPATH='/mnt/lustre/share_data/zhujinguo/data/bert_pretrain_data/wikitext'

# wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt

fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref $DATAPATH/wikitext-103-raw/wiki.train.bpe \
    --validpref $DATAPATH/wikitext-103-raw/wiki.valid.bpe \
    --testpref $DATAPATH/wikitext-103-raw/wiki.test.bpe \
    --destdir $DATAPATH/wikitext-103-raw/data-bin/wikitext-103 \
    --workers 60