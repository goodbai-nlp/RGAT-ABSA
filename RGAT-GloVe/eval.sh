#!/bin/bash

emb_dir=/mnt/data2/xfbai/data/embeddings/glove

CUDA_VISIBLE_DEVICES=0 python -u eval.py \
    --pretrained_model_path $1 \
	--data_dir $2 \
	--vocab_dir $2 \
	--glove_dir $emb_dir