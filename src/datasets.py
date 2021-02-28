import os
import pathlib

import numpy as np
import pandas as pd
import tqdm
import pickle
from gensim.models import KeyedVectors


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
    
def read_allocine(path="datasets/data" , split="train"):
    """
    Get the allocine dataset for classification.
    split is either train, test or val
    """
    
    assert split in ['train', 'val', 'test']
    
    with (open("datasets/data/allocine_dataset.pickle", "rb")) as f:
        data = pickle.load(f)
        
    texts = list(data[split + '_set']['review'])
    labels = list(data[split + '_set']['polarity'])
    
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
    
    
def read_french_w2v(path='datasets/frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin', lim=100000):
    """
    Read word 2 vec into a pandas DataFrame
    Limite the total number of words for performance issues
    """
    
    model = KeyedVectors.load_word2vec_format(path, binary=True, unicode_errors="ignore")
    
    list_of_ranked_words = model.index2word
    voc_size = len(list_of_ranked_words)
    nb_words = min(lim, voc_size)
    print(nb_words)
    dims = model.word_vec('singe').shape[0]
    
    words = []
    vectors = np.zeros((nb_words, dims), dtype=np.float32)
    
    for k in tqdm.tqdm(range(nb_words)):
        w = list_of_ranked_words[k]
        vec = model.word_vec(w)
        vectors[k] = list(vec)
        words.append(w)
    return pd.DataFrame(vectors, index=words)
    
    
    
    
    
    
    
    
    
    
    

