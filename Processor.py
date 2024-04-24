from DataReader import DataReader
from Preprocessor import Preprocessor
from Vectorizer import Vectorizer
from Classifier import Classifier
from BERTClassifier import bert_classifier
from Plotter import plot as plt
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import f1_score
import nltk
import warnings

warnings.filterwarnings('ignore')


def train(filepath, preprocessors, vectorizers, classifiers):
    nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger'])
    dr = DataReader(filepath, 'A')
    data, labels = dr.get_labelled_data()
    # data, labels = dr.shuffle(data, labels, 'random')
    tr_data, tst_data, tr_labels, tst_labels = split(data, labels, test_size=0.3)
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
    accs = []
    f1_scores = []
    classifier_names = []
    classifier_params = []
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
        classifier_params.append(clf.params)

    return tr_data, tst_data, classifier_names, classifier_params, accs, f1_scores, dr


def train_without_bert(filepath, preprocessors, vectorizers, classifiers):
    tr_data, tst_data, classifier_names, classifier_params, accs, f1_scores, _ = train(filepath, preprocessors,
                                                                                       vectorizers, classifiers)

    plt(filepath, tr_data, tst_data, classifier_names, classifier_params, accs, f1_scores)


def train_with_bert(filepath, preprocessors, vectorizers, classifiers, batch_sizes):
    tr_data, tst_data, classifier_names, classifier_params, accs, f1_scores, dr = train(filepath, preprocessors,
                                                                                        vectorizers, classifiers)

    for batch_size in batch_sizes:
        bert_acc, bert_f1 = bert_classifier(batch_size, dr)
        print(f"Accuracy: {bert_acc}, F1 Score: {bert_f1}, Classifier: BERT, Params: {{'batch_size': {batch_size}}}")
        accs.append(bert_acc)
        f1_scores.append(bert_f1)
        classifier_names.append('BERT')
        classifier_params.append({'batch_size': batch_size})

    plt(filepath, tr_data, tst_data, classifier_names, classifier_params, accs, f1_scores)
