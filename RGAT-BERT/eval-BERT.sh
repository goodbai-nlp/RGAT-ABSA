#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 -u bert_eval.py \
    --pretrained_model_path $1 \
    --data_dir $2 \
	--vocab_dir $2 \
