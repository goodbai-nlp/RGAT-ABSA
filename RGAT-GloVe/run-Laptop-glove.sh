#!/bin/bash
# training command for different datasets.
source_dir=../dataset
emb_dir=/mnt/data2/xfbai/data/embeddings/glove
save_dir=saved_models

exp_setting=train

####### Laptops  acc:78.02 f1:74.00 #########
exp_dataset=Biaffine/glove/Laptops
exp_path=$save_dir/Laptops/$exp_setting
if [ ! -d "$exp_path" ]; then
  mkdir -p "$exp_path"
fi

CUDA_VISIBLE_DEVICES=0 python -u train.py \
	--data_dir $source_dir/$exp_dataset \
	--vocab_dir $source_dir/$exp_dataset \
	--glove_dir $emb_dir \
	--model "RGAT" \
	--save_dir $exp_path \
	--seed 48 \
	--pooling "avg" \
	--output_merge "gate" \
	--num_layers 4 \
	--attn_heads 5 \
	--num_epoch 60 \
	--shuffle \
	2>&1 | tee $exp_path/training.log
