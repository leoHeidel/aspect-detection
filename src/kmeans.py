# kmeans.py

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans

class KMeansAspectDetector:
    """
    Detect aspect by applying K means to the sets of word vectors.
    """
    def __init__(self, w2v, dataset = None, k = 5, language="english", dist="L2"):
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
        self.dataset = dataset
        self.k_means = None
        self.raw_aspects = None
        
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
        return vectors

    
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
        vectors = self.transform_sentence(sentence)
        k_means = KMeans(self.k).fit(vectors)
        aspects_vectors = k_means.cluster_centers_
        
        aspects = []
        for i in range(self.k):
            aspect = self._retrieve_word(aspects_vectors[i])
            aspects.append(aspect)
        return aspects 
    
    def create_dataset(self):
        
        N_data = len(self.dataset)
        
        dfs = []
        for idx, review in enumerate(self.dataset):
            dfs.append(self.transform_sentence(review))
        df = pd.concat(dfs)#.astype('int32')
        print("df shape ", df.shape)
        print(df.head())
        return df
        
           
    def train(self):
        """
        Predict aspects for single sentence.
        """
        
        print("Training Started !")
        
        data = self.create_dataset()
        
        self.k_means = KMeans(self.k).fit(data)
        self.raw_aspects = self.k_means.cluster_centers_
        
        print("Done !")
        
        
    def predict_aspect(self):
        
        aspects_vectors = self.raw_aspects
        
        aspects = []
        for i in range(self.k):
            aspect = self._retrieve_word(aspects_vectors[i])
            aspects.append(aspect)
        return aspects























