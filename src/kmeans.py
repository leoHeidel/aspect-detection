# kmeans.py

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans

class KMeansAspectDetector:
    """
    Detect aspect by applying K means to the sets of word vectors.
    """
    def __init__(self, w2v, k = 5, language="english", dist="L2"):
        """
        k : the number of predicted aspects
        dist : L2 or cosin, distance to use to retrive a word from a vector
        """
        self.stop_words = set(stopwords.words(language))
        valid_words = [word for word in w2v.index if word not in self.stop_words and word.isalnum()]
        self.w2v = w2v.loc[valid_words]
        self.w2v_words = set(valid_words)
        self.k = k
        self.dist = dist
        
    def transform_sentence(self, sentence):
        """
        Transform single sentence to vectors of aspects.
        Removes stop words and punctuation.
        Apply Kmeans to predicts aspect raw vectors (No necessarily exact words) 
        """
        word_tokens = word_tokenize(sentence)  
        filtered_sentence = [w for w in word_tokens if not w in self.stop_words]  
        filtered_sentence = [word for word in filtered_sentence if word.isalnum()]
        filtered_sentence = [word for word in filtered_sentence if word in self.w2v_words]
        vectors = self.w2v.loc[filtered_sentence]
        k_means = KMeans(self.k).fit(vectors)
        raw_aspects = k_means.cluster_centers_
        return raw_aspects
    
    def _retrieve_word(self, aspects_vector):
        """
        Retrieve word closest to a vector
        """
        if self.dist == "cosin":
            scores = (self.w2v @ aspects_vector)
        else:
            assert self.dist == "L2", f"Unknown distance {self.dist}"
            diff = self.w2v - aspects_vector
            scores = -(diff*diff).sum(axis=1)
            
        return scores.index[scores.argmax()]
    
    def predict_sentence(self, sentence):
        """
        Predict aspects for single sentence.
        """
        aspects_vectors = self.transform_sentence(sentence)
        aspects = []
        for i in range(self.k):
            aspect = self._retrieve_word(aspects_vectors[i])
            aspects.append(aspect)
        return aspects 