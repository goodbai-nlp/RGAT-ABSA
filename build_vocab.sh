#!/bin/bash
# build vocab for different datasets
python prepare_vocab.py --data_dir dataset/Restaurants --vocab_dir dataset/Restaurants
python prepare_vocab.py --data_dir dataset/Laptops --vocab_dir dataset/Laptops
python prepare_vocab.py --data_dir dataset/Tweets --vocab_dir dataset/Tweets
python prepare_vocab.py --data_dir dataset/MAMS --vocab_dir dataset/MAMS
