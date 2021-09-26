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

   + Downloading Glove embeddings (available at [here](http://nlp.stanford.edu/data/glove.840B.300d.zip)), then  run 

     ```
     awk '{print $1}' glove.840B.300d.txt > glove_words.txt
     ```

     to get `glove_words.txt`.

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


5. Evaluating trained models (optional)

   ``` 
   bash eval.sh path/to/check_point path/to/dataset
   bash eval-BERT.sh path/to/check_point path/to/dataset
   ```
## Results

### GloVe-based Model

|Setting|  Acc  | F1  | Log | Pretrained model |
|  :----:  | :----:  |:---:|  :----:  | :----:  |
| Res14  | 83.55 | 75.99 | [log](https://github.com/muyeby/RGAT-ABSA/blob/master/RGAT-GloVe/saved_models/Restaurants/train/training.log) | [model](https://drive.google.com/file/d/1IIVsRUjSZiYUEjv0hOVyl3AS4FAnCmdG/view?usp=sharing) |
| Laptop  | 78.02 | 74.00 | [log](https://github.com/muyeby/RGAT-ABSA/blob/master/RGAT-GloVe/saved_models/Laptops/train/training.log) | [model](https://drive.google.com/file/d/1Q1MHf8vDUqmhb3w7m4stpg3hyyig9dvl/view?usp=sharing) |
| Tweets  | 75.37 | 74.15 | [log](https://github.com/muyeby/RGAT-ABSA/blob/master/RGAT-GloVe/saved_models/Tweets/train/training.log) | [model](https://drive.google.com/file/d/1Ma1DXlNeb09CJqVd_4i-4oXpBmEElBzJ/view?usp=sharing) |
| MAMS  | 82.02 | 80.99 | [log](https://github.com/muyeby/RGAT-ABSA/blob/master/RGAT-GloVe/saved_models/MAMS/train/training.log) | [model](https://drive.google.com/file/d/1ofVXcyhrvkAPbA8HXn7wXErN2uany-Mv/view?usp=sharing) |

### BERT-based Model

|Setting|  Acc  | F1  | Log | Pretrained model |
|  :----:  | :----:  |:---:|  :----:  | :----:  |
| Res14  | 86.68 | 80.92 | [log](https://github.com/muyeby/RGAT-ABSA/blob/master/RGAT-BERT/saved_models/Restaurants/train/training.log) | [model](https://drive.google.com/file/d/1P9K8yu6nccbxIu2vc2ZOvu16m4ggFzlG/view?usp=sharing) |
| Laptop  | 82.34 | 78.94 | [log](https://github.com/muyeby/RGAT-ABSA/blob/master/RGAT-BERT/saved_models/Laptops/train/training.log) | [model](https://drive.google.com/file/d/122R8sthFFLQZjkCqc7unFsyGZ2h9t_hk/view?usp=sharing)  |
| Tweets  | 76.28 | 75.41 | [log](https://github.com/muyeby/RGAT-ABSA/blob/master/RGAT-BERT/saved_models/Tweets/train/training.log) | [model](https://drive.google.com/file/d/14oMqTjAO11Jws5wyMiT95NCOUWaUwkDX/view?usp=sharing)  |
| MAMS  | 84.52 | 83.74 | [log](https://github.com/muyeby/RGAT-ABSA/blob/master/RGAT-BERT/saved_models/MAMS/train/training.log) | [model](https://drive.google.com/file/d/1Arzpzj3xnxsCnOb0IETUpqTpEKofpcnJ/view?usp=sharing)  |


## References

```
@ARTICLE{bai21syntax,  
	author={Xuefeng Bai and Pengbo Liu and Yue Zhang},  
	journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},   
	title={Investigating Typed Syntactic Dependencies for Targeted Sentiment Classification Using Graph Attention Neural Network},   
	year={2021},  
	volume={29}, 
	pages={503-514},  
	doi={10.1109/TASLP.2020.3042009}
}
```



