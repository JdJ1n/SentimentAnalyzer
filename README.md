# Offenseval
Based on SemEval 2019 - Task 6 - Identifying and Categorizing Offensive Language in Social Media by ZeyadZanaty

## Descrption
A system to classifiy a tweet as either offensive or not offensive (Sub-task A). Some sort of grid search approach is taken where multiple techniques for preprocessing, feature extraction and classification are implemented and combinations of them all are tried to achieve the best model for the given dataset.

## Subtasks
- Sub-task A - Offensive language identification;  [Offensive: OFF, Not Offensive: NOT]

## Implementation

### Preprocessing
Tokenization, Stopwords Removal, Stemming

### Vectorization
TFIDF, Count, Word2Vec, GloVe, fastText

### Classification
Naïve Bayes, Decision Trees, Random Forest, SVM, MLP
#### Deeplearning
BERT, LSTM, 1-D CNN

## Running
- Install requiremetns using `pip3 install -r requirements.txt`
 
- `python3 tune.py` to do a complete search on all combinations of prepocessing, vectorization and non-deep classification techniques while tuning the classifiers hyper-params.

- `python3 train.py` to train one of the deeplearning models.

## Sample tuning.py output
![alt text](https://github.com/ZeyadZanaty/offenseval/blob/master/docs/tuning-reults/tuning-b-f1/LogisticRegression.png?raw=true "Logistic Regression")
![alt text](https://github.com/ZeyadZanaty/offenseval/blob/master/docs/tuning-reults/tuning-b-f1/SVC.png?raw=true "SVC")
![alt text](https://github.com/ZeyadZanaty/offenseval/blob/master/docs/tuning-reults/tuning-a/RandomForest.png?raw=true "RF")
