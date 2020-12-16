# Investigating Typed Syntactic Dependencies for Targeted Sentiment Classification Using Graph Attention Neural Network
This repository contains the code for the paper "Investigating Typed Syntactic Dependencies for Targeted Sentiment Classification Using Graph Attention Neural Network", IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)

## Setup

This code runs Python 3.6 with the following libraries:

+ Pytorch 1.2.0
+ Torchtext 0.4.0
+ GTX 1080 Ti

## Get start

1. Prepare data
   + Restaurants, Laptop and Tweets dataset. (We provide the parsed data at directory `dataset`)
   + Glove embeddings (available at [here](http://nlp.stanford.edu/data/glove.840B.300d.zip))

2. Build vocabulary

   ```
   bash build_vocab.sh
   ```

3. Training

   ``` 
   bash run-res-glove.sh
   bash run-laptop-glove.sh
   bash run-tweets-glove.sh
   ```

4. The saved model and training logs will be stored at directory `saved_models`  

## References

Unavailable now.
