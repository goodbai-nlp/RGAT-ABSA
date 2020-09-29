#!/bin/bash
# build vocab for different datasets
#catelst=("Biaffine/glove" "Biaffine/bert" "Lexical")
catelst=("Biaffine/glove" "Biaffine/bert")
for cate in ${catelst[@]}
do
	echo $cate
	python prepare_vocab.py --data_dir dataset/$cate/Restaurants --vocab_dir dataset/$cate/Restaurants
	python prepare_vocab.py --data_dir dataset/$cate/Laptops --vocab_dir dataset/$cate/Laptops
	python prepare_vocab.py --data_dir dataset/$cate/Tweets --vocab_dir dataset/$cate/Tweets
done
