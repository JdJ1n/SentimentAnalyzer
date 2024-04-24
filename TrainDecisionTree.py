from Pivot import train_without_bert as train

filepath = 'datasets/offenseval-training-v3.tsv'

preprocessors = [('remove_stopwords', 'stem'), ('remove_stopwords',), ('stem',),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem')]

vectorizers = ['count', 'count', 'count', 'count', 'count', 'count', 'count', 'tfidf', 'tfidf', 'tfidf', 'tfidf']

classifiers = [
    ('Dummy', {'strategy': 'uniform'}),
    ('DecisionTree', {'criterion': 'gini', 'min_samples_split': 2}),
    ('DecisionTree', {'criterion': 'gini', 'min_samples_split': 2}),
    ('DecisionTree', {'criterion': 'gini', 'min_samples_split': 2}),
    ('DecisionTree', {'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 2}),
    ('DecisionTree', {'criterion': 'entropy', 'min_samples_split': 2}),
    ('DecisionTree', {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 2}),
    ('DecisionTree', {'criterion': 'gini', 'min_samples_split': 2}),
    ('DecisionTree', {'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 2}),
    ('DecisionTree', {'criterion': 'entropy', 'min_samples_split': 2}),
    ('DecisionTree', {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 2}),
]

train(filepath, preprocessors, vectorizers, classifiers)
