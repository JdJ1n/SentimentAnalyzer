from Processor import train_without_bert as train

filepath = 'datasets/offenseval-training-v1.tsv'

preprocessors = [('remove_stopwords', 'stem'), ('remove_stopwords',), ('stem',),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem')]

vectorizers = ['count', 'count', 'count', 'count', 'count', 'count', 'count', 'tfidf', 'tfidf', 'tfidf', 'tfidf']

classifiers = [
    ('Dummy', {'strategy': 'uniform'}),
    ('M-NaiveBayes', {'alpha': 5, 'fit_prior': True}),
    ('M-NaiveBayes', {'alpha': 5, 'fit_prior': True}),
    ('M-NaiveBayes', {'alpha': 5, 'fit_prior': True}),
    ('M-NaiveBayes', {'alpha': 10, 'fit_prior': True}),
    ('G-NaiveBayes', {'var_smoothing': 1e-9}),
    ('G-NaiveBayes', {'var_smoothing': 1e-18}),
    ('M-NaiveBayes', {'alpha': 5, 'fit_prior': True}),
    ('M-NaiveBayes', {'alpha': 10, 'fit_prior': True}),
    ('G-NaiveBayes', {'var_smoothing': 1e-9}),
    ('G-NaiveBayes', {'var_smoothing': 1e-18}),
]

train(filepath, preprocessors, vectorizers, classifiers)
