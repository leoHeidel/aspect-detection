import src

def test_imdb():
    """
    If fails, verify that imdb has been properly downloaded
    """
    train_texts, train_labels = src.datasets.read_imdb()    
    test_texts, test_labels = src.datasets.read_imdb(split="train")
    assert len(train_texts) == len(train_labels)    
    assert len(test_texts) == len(test_labels)
    
    
def test_english_w2v():
    df = src.datasets.read_english_w2v(lim=10000)
    assert not df.isnull().values.any()