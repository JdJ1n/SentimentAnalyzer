from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings('ignore', category=DeprecationWarning)


class Vectorizer:

    def __init__(self, typename, pre_trained=False, retrain=False, extend_training=False, params=None):
        self.vocab_length = None
        self.vectors = None
        self.data = None
        if params is None:
            params = {}
        self.type = typename
        self.pre_trained = pre_trained
        self.params = params
        self.retrain = retrain
        self.extend_training = extend_training
        self.vectorizer = None
        self.max_len = None

    def tfidf(self):
        vectorizer = TfidfVectorizer(**self.params)
        untokenized_data = [' '.join(tweet) for tweet in self.data]
        if not self.vectorizer:
            self.vectorizer = vectorizer.fit(untokenized_data)
        self.vectors = self.vectorizer.transform(untokenized_data).toarray()
        return self.vectors

    def count(self):
        vectorizer = CountVectorizer(**self.params)
        untokenized_data = [' '.join(tweet) for tweet in self.data]
        if not self.vectorizer:
            self.vectorizer = vectorizer.fit(untokenized_data)
        self.vectors = self.vectorizer.transform(untokenized_data).toarray()
        self.vocab_length = len(self.vectorizer.vocabulary_.keys())
        return self.vectors

    def vectorize(self, data):
        self.data = data
        vectorize_call = getattr(self, self.type, None)
        if vectorize_call:
            vectorize_call()
        else:
            raise Exception(str(self.type), 'is not an available function')
        return self.vectors

    def fit(self, data):
        self.data = data
