#!/bin/bash
# build vocab for different datasets
setting=Biaffine/glove
python prepare_vocab.py --data_dir dataset/$setting/Restaurants --vocab_dir dataset/$setting/Restaurants
python prepare_vocab.py --data_dir dataset/$setting/Laptops --vocab_dir dataset/$setting/Laptops
python prepare_vocab.py --data_dir dataset/$setting/Tweets --vocab_dir dataset/$setting/Tweets
python prepare_vocab.py --data_dir dataset/$setting/MAMS --vocab_dir dataset/$setting/MAMS
