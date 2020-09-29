#!/bin/bash
# training command for different datasets.
source_dir=dataset
emb_dir=/mnt/data2/xfbai/data/embeddings/glove
save_dir=saved_models
layers=6
exp_model="RGAT"
seed=14
nheads=10
merge="None"
pool="avg"

exp_setting=${layers}layer-${nheads}head-${pool}-${merge}

####### Restaurants  acc:83.55 f1:75.99 #########
exp_dataset=Biaffine/glove/Restaurants
exp_path=$save_dir/$exp_dataset/$exp_model/$exp_setting
if [ ! -d "$exp_path" ]; then
  mkdir -p "$exp_path"
fi

CUDA_VISIBLE_DEVICES=0 python -u train.py \
	--data_dir $source_dir/$exp_dataset \
	--glove_dir $emb_dir \
	--model $exp_model \
	--exp_id $exp_dataset/$exp_model/$exp_setting \
	--vocab_dir $source_dir/$exp_dataset \
	--save_dir $save_dir \
	--seed 14 \
	--pooling "avg" \
	--output_merge "None" \
	--num_layers 6 \
	--attn_heads 10 \
	--num_epoch 30 \
	--shuffle \
	2>&1 | tee $exp_path/training.log
