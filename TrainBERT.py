from Pivot import train_with_bert as train

filepath = 'datasets/offenseval-training-v2.tsv'

preprocessors = [('remove_stopwords', 'stem')]

vectorizers = ['count']

classifiers = [('Dummy', {'strategy': 'uniform'})]

batch_sizes = [20, 40, 60, 80, 100, 120]

train(filepath, preprocessors, vectorizers, classifiers, batch_sizes)
