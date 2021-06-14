#!/bin/bash

source_dir=../dataset
save_dir=saved_models

exp_setting=train
exp_dataset=Biaffine/glove/Laptops

############# Laptops acc:82.34, f1:78.94 #################
exp_path=$save_dir/Laptops/$exp_setting

if [ ! -d "$exp_path" ]; then
  echo "making new dir.."
  mkdir -p "$exp_path"
fi

CUDA_VISIBLE_DEVICES=0 python -u bert_train.py \
	--lr 1e-5 \
	--bert_lr 2e-5 \
	--input_dropout 0.1 \
	--att_dropout 0.1 \
	--num_layer 2 \
	--bert_out_dim 100 \
	--dep_dim 100 \
	--max_len 90 \
	--data_dir $source_dir/$exp_dataset \
	--vocab_dir $source_dir/$exp_dataset \
	--save_dir $exp_path \
	--model "RGAT" \
	--seed 9 \
	--output_merge "gate" \
	--reset_pool \
	--num_epoch 15 2>&1 | tee $exp_path/training.log
