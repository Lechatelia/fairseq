#!/bin/bash
# DATAPATH='/nfs/zhujinguo/datasets/data/bert_pretrain_data/bookswiki'
DATAPATH='/mnt/lustre/share_data/zhujinguo/data/bert_pretrain_data/bookswiki'

# wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt

srun -p cpu \
fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref $DATAPATH/bookswiki.bpe \
    --validpref $DATAPATH/bookswiki-1000.bpe \
    --testpref $DATAPATH/bookswiki-1000.bpe \
    --destdir $DATAPATH/data-bin/bookswiki-combine \
    --workers 60
