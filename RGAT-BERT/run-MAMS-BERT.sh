#!/usr/bin/bash

source_dir=../dataset
save_dir=saved_models

exp_setting=train
exp_dataset=Biaffine/glove/MAMS

############# MAMS acc:84.52 f1:83.74 #################

exp_path=$save_dir/MAMS/$exp_setting
if [ ! -d "$exp_path" ]; then
  mkdir -p "$exp_path"
fi
CUDA_VISIBLE_DEVICES=0 python3 -u bert_train.py \
	--lr 1e-5 \
	--bert_lr 2e-5 \
	--input_dropout 0.1 \
	--att_dropout 0.0 \
	--num_layer 2 \
	--bert_out_dim 100 \
	--dep_dim 80 \
	--max_len 90 \
	--data_dir $source_dir/$exp_dataset \
	--vocab_dir $source_dir/$exp_dataset \
	--save_dir $exp_path \
	--model "RGAT" \
	--seed 38 \
	--output_merge "gate" \
	--reset_pool \
	--num_epoch 10 2>&1 | tee $exp_path/training.log
