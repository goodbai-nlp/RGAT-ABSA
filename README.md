# Investigating Typed Syntactic Dependencies for Targeted Sentiment Classification Using Graph Attention Neural Network

This repository contains the code for the paper "[Investigating Typed Syntactic Dependencies for Targeted Sentiment Classification Using Graph Attention Neural Network](https://arxiv.org/abs/2002.09685)", IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)

## Setup

This code runs Python 3.6 with the following libraries:

+ Pytorch 1.2.0
+ Transformers 2.9.1
+ GTX 1080 Ti

You can also create an virtual environments with `conda` by run

```
conda env create -f requirements.yaml
```

## Get start

1. Prepare data

   + Restaurants, Laptop, Tweets and MAMS dataset. (We provide the parsed data at directory `dataset`)
   + Glove embeddings (available at [here](http://nlp.stanford.edu/data/glove.840B.300d.zip))

2. Build vocabulary

   ```
   bash build_vocab.sh
   ```

3. Training
   Go to Corresponding directory and run scripts:

   ``` 
   bash run-MAMS-glove.sh
   bash run-MAMS-BERT.sh
   ```

4. The saved model and training logs will be stored at directory `saved_models`  

## References

Unavailable now.
