#!/bin/bash
# training command for different datasets.
source_dir=../dataset
emb_dir=/mnt/data2/xfbai/data/embeddings/glove
save_dir=saved_models

exp_setting=train

####### MAMS  acc:82.02 f1:80.99 #########
exp_dataset=Biaffine/glove/MAMS
exp_path=$save_dir/MAMS/$exp_setting
if [ ! -d "$exp_path" ]; then
  mkdir -p "$exp_path"
fi

CUDA_VISIBLE_DEVICES=1 python -u train.py \
	--data_dir $source_dir/$exp_dataset \
	--vocab_dir $source_dir/$exp_dataset \
	--glove_dir $emb_dir \
	--model "RGAT" \
	--save_dir $exp_path \
	--seed 14 \
	--batch_size 32 \
	--pooling "avg" \
	--output_merge "gate" \
	--num_layers 4 \
	--attn_heads 5 \
	--num_epoch 65 \
	--shuffle \
	2>&1 | tee $exp_path/training.log
