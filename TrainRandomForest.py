from Processor import train_without_bert as train

filepath = 'datasets/offenseval-training-v1.tsv'

preprocessors = [('remove_stopwords', 'stem'), ('remove_stopwords',), ('stem',),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem')]

vectorizers = ['count', 'count', 'count', 'count', 'count', 'count', 'count', 'tfidf', 'tfidf', 'tfidf', 'tfidf']

classifiers = [
    ('Dummy', {'strategy': 'uniform'}),
    ('RandomForest', {'n_estimators': 30, 'max_depth': 5}),
    ('RandomForest', {'n_estimators': 30, 'max_depth': 5}),
    ('RandomForest', {'n_estimators': 30, 'max_depth': 5}),
    ('RandomForest', {'n_estimators': 100, 'max_depth': 5}),
    ('RandomForest', {'n_estimators': 30, 'max_depth': 10}),
    ('RandomForest', {'n_estimators': 100, 'max_depth': 10}),
    ('RandomForest', {'n_estimators': 30, 'max_depth': 5}),
    ('RandomForest', {'n_estimators': 100, 'max_depth': 5}),
    ('RandomForest', {'n_estimators': 30, 'max_depth': 10}),
    ('RandomForest', {'n_estimators': 100, 'max_depth': 10}),
]

train(filepath, preprocessors, vectorizers, classifiers)
