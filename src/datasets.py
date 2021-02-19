import os
import pathlib

import numpy as np
import pandas as pd
import tqdm


def read_imdb(path="datasets/aclImdb" , split="train"):
    """
    Get the IMDB dataset for classification.
    split is either train or test
    """
    split_path = os.path.join(path, split)
    split_dir = pathlib.Path(split_path)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels



def read_english_w2v(path="datasets/wiki-news-300d-1M.vec", lim=100000):
    """
    Read word 2 vec into a pandas DataFrame
    Limite the total number of words for performance issues
    """
    with open(path) as f:
        nb_words, dims = [int(d) for d in f.readline().split()]
        nb_words = min(lim, nb_words)
        words = []
        vectors = np.zeros((nb_words, dims), dtype=np.float32)
        current = 0
        
        for l in tqdm.tqdm(f, total=nb_words):
            idx = l.index(" ")
            w = l[:idx]
            words.append(w)
            vec = np.fromstring(l[idx :], sep=' ', dtype=np.float32)
            vectors[current] = vec
            current += 1
            if current >= nb_words:
                break
    return pd.DataFrame(vectors, index=words)

