from DataReader import DataReader
from Preprocessor import Preprocessor
from Vectorizer import Vectorizer
from Classifier import Classifier
from BERTClassifier import bert_classifier
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import f1_score
import numpy as np
import nltk
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger'])

dr = DataReader('datasets/offenseval-training-v1.tsv', 'A')
data, labels = dr.get_labelled_data()
data, labels = dr.shuffle(data, labels, 'random')

tr_data, tst_data, tr_labels, tst_labels = split(data, labels, test_size=0.3)

preprocessors = [('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem'),
                 ('remove_stopwords', 'stem'), ('remove_stopwords', 'stem')]

vectorizers = ['count', 'count', 'count', 'count', 'count', 'tfidf', 'tfidf', 'tfidf', 'tfidf']

classifiers = [
    ('Dummy', {'strategy': 'uniform'}),
    ('SVC', {'C': 1, 'kernel': 'linear'}),
    ('SVC', {'C': 10, 'kernel': 'linear'}),
    ('SVC', {'C': 1, 'kernel': 'rbf'}),
    ('SVC', {'C': 10, 'kernel': 'rbf'}),
    ('SVC', {'C': 1, 'kernel': 'linear'}),
    ('SVC', {'C': 10, 'kernel': 'linear'}),
    ('SVC', {'C': 1, 'kernel': 'rbf'}),
    ('SVC', {'C': 10, 'kernel': 'rbf'}),
]

for i in range(len(classifiers)):
    preprocessors[i] = Preprocessor(preprocessors[i])
    vectorizers[i] = Vectorizer(vectorizers[i])
    classifiers[i] = Classifier(*classifiers[i])

tst_vecs = []
tr_vecs = []
for i in range(len(classifiers)):
    tr_data_clean = preprocessors[i].clean(tr_data)
    tst_data_clean = preprocessors[i].clean(tst_data)

    tr_vecs.append(vectorizers[i].vectorize(tr_data_clean))
    tst_vecs.append(vectorizers[i].vectorize(tst_data_clean))

    classifiers[i].fit(tr_vecs[i], tr_labels)

plt.ion()  # 开启交互模式

accs = []
f1_scores = []
classifier_names = []
for i, clf in enumerate(classifiers):
    acc = clf.test_and_plot(tst_vecs[i], tst_labels, class_num=2)
    accs.append(acc)

    # Calculate F1 score
    pred = clf.predict(tst_vecs[i])
    f1 = f1_score(tst_labels, pred, average='weighted')
    f1_scores.append(f1)

    print(f"Accuracy: {acc}, F1 Score: {f1}, Classifier: {clf.classifier.__name__}, Params: {clf.params}")

    # Append classifier name for the plot
    classifier_names.append(clf.classifier.__name__)

# Ajouter des data de BERT
# bert_batch_size=40
# bert_acc, bert_f1=bert_classifier(bert_batch_size,dr)
# print(f"Accuracy: {bert_acc}, F1 Score: {bert_f1}, Classifier: BERT, Params: 'batch_size': {bert_batch_size}")
#
# # Add BERT data to the lists
# accs.append(bert_acc)
# f1_scores.append(bert_f1)
# classifier_names.append('BERT')
# bert_params = {'batch_size': bert_batch_size}

# Plotting
x = np.arange(len(classifier_names))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width / 2, accs, width, label='Accuracy')
rects2 = ax.bar(x + width / 2, f1_scores, width, label='F1 Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title(f'Scores by classifier (Train size: {len(tr_data)}, Test size: {len(tst_data)})')
ax.set_xticks(x)

# Add classifier parameters to x-axis labels
ax.set_xticklabels([f'{name}\\nParams: {params}' for name, params in zip(classifier_names, [clf.params for clf in classifiers])], wrap=True, fontsize=8)  # Update this line
plt.xticks(rotation=45)

ax.legend()

fig.tight_layout()

plt.show(block=True)
