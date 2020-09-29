#!/bin/bash
# training command for different datasets.
source_dir=dataset
emb_dir=/mnt/data2/xfbai/data/embeddings/glove
save_dir=saved_models

exp_setting=${layers}layer-${nheads}head-${pool}-${merge}

####### Tweets  acc:75.36 f1:74.15 #########
exp_dataset=Biaffine/glove/Tweets
exp_path=$save_dir/$exp_dataset/$exp_model/$exp_setting
if [ ! -d "$exp_path" ]; then
  mkdir -p "$exp_path"
fi

CUDA_VISIBLE_DEVICES=1 python -u train.py \
	--data_dir $source_dir/$exp_dataset \
	--glove_dir $emb_dir \
	--model "RGAT" \
	--exp_id $exp_dataset/$exp_model/$exp_setting \
	--vocab_dir $source_dir/$exp_dataset \
	--save_dir $save_dir \
	--seed 22 \
	--pooling "avg" \
	--output_merge "gate" \
	--num_layers 6 \
	--attn_heads 5 \
	--num_epoch 60 \
	--shuffle \
	2>&1 | tee $exp_path/training.log
