from Processor import train_without_bert as train

filepath = 'datasets/offenseval-training-v4.tsv'

preprocessors = [('remove_stopwords', 'stem'), ('remove_stopwords',), ('stem',),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'), ]

vectorizers = ['count', 'count', 'count', 'count', 'count', 'count', 'count', 'count',
               'count', 'tfidf', 'tfidf', 'tfidf', 'tfidf', 'tfidf', 'tfidf']

classifiers = [
    ('Dummy', {'strategy': 'uniform'}),
    ('MLP', {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam'}),
    ('MLP', {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam'}),
    ('MLP', {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam'}),
    ('MLP', {'hidden_layer_sizes': (100,), 'activation': 'logistic', 'solver': 'adam'}),
    ('MLP', {'hidden_layer_sizes': (200,), 'activation': 'relu', 'solver': 'adam'}),
    ('MLP', {'hidden_layer_sizes': (200,), 'activation': 'logistic', 'solver': 'adam'}),
    ('MLP', {'hidden_layer_sizes': (500,), 'activation': 'relu', 'solver': 'adam'}),
    ('MLP', {'hidden_layer_sizes': (500,), 'activation': 'logistic', 'solver': 'adam'}),
    ('MLP', {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam'}),
    ('MLP', {'hidden_layer_sizes': (100,), 'activation': 'logistic', 'solver': 'adam'}),
    ('MLP', {'hidden_layer_sizes': (200,), 'activation': 'relu', 'solver': 'adam'}),
    ('MLP', {'hidden_layer_sizes': (200,), 'activation': 'logistic', 'solver': 'adam'}),
    ('MLP', {'hidden_layer_sizes': (500,), 'activation': 'relu', 'solver': 'adam'}),
    ('MLP', {'hidden_layer_sizes': (500,), 'activation': 'logistic', 'solver': 'adam'}),
]

train(filepath, preprocessors, vectorizers, classifiers)
