import linecache
import numpy as np

# load pretrained word embeddings


def load_pretrained_embedding(glove_dir, word_list, dimension_size=300, encoding="utf-8"):
    pre_words = []
    count = 0

    with open(glove_dir + "/glove_words.txt", "r", encoding=encoding) as fopen:
        for line in fopen:
            pre_words.append(line.strip())
    word2offset = {w: i for i, w in enumerate(pre_words)}

    word_vectors = []
    for word in word_list:
        if word in word2offset:
            line = linecache.getline(glove_dir + "/glove.840B.300d.txt", word2offset[word] + 1)
            assert word == line[: line.find(" ")].strip()
            word_vectors.append(
                np.fromstring(line[line.find(" "):].strip(), sep=" ", dtype=np.float32)
            )
            count += 1
        else:
            # init zero
            word_vectors.append(np.zeros(dimension_size, dtype=np.float32))
    print("Loading {}/{} words from vocab...".format(count, len(word_list)))

    return word_vectors
