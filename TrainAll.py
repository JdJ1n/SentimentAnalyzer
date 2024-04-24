from Processor import train_with_bert as train

filepath = 'datasets/offenseval-training-v4.tsv'

preprocessors = [('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem')]

vectorizers = ['count', 'count', 'count', 'count', 'count', 'tfidf']

classifiers = [
    ('Dummy', {'strategy': 'uniform'}),
    ('M-NaiveBayes', {'alpha': 5, 'fit_prior': True}),
    ('DecisionTree', {'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 2}),
    ('RandomForest', {'n_estimators': 30, 'max_depth': 10}),
    ('SVC', {'C': 1, 'kernel': 'linear'}),
    ('MLP', {'hidden_layer_sizes': (500,), 'activation': 'relu', 'solver': 'adam'})
]

batch_sizes = [60]

train(filepath, preprocessors, vectorizers, classifiers, batch_sizes)
