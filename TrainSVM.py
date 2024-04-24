from Processor import train_without_bert as train

filepath = 'datasets/offenseval-training-v3.tsv'

preprocessors = [('remove_stopwords', 'stem'), ('remove_stopwords',), ('stem',),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem')]

vectorizers = ['count', 'count', 'count', 'count', 'count', 'count', 'count', 'tfidf', 'tfidf', 'tfidf', 'tfidf']

classifiers = [
    ('Dummy', {'strategy': 'uniform'}),
    ('SVC', {'C': 1, 'kernel': 'linear'}),
    ('SVC', {'C': 1, 'kernel': 'linear'}),
    ('SVC', {'C': 1, 'kernel': 'linear'}),
    ('SVC', {'C': 10, 'kernel': 'linear'}),
    ('SVC', {'C': 1, 'kernel': 'rbf'}),
    ('SVC', {'C': 10, 'kernel': 'rbf'}),
    ('SVC', {'C': 1, 'kernel': 'linear'}),
    ('SVC', {'C': 10, 'kernel': 'linear'}),
    ('SVC', {'C': 1, 'kernel': 'rbf'}),
    ('SVC', {'C': 10, 'kernel': 'rbf'}),
]

train(filepath, preprocessors, vectorizers, classifiers)
