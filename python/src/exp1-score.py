# coding: utf-8
#
#

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix

LAMBDA = 0.01


def load_score_data(trn_fn, tst_fn):
    train_df = pd.read_csv(trn_fn, sep='\t', header=None, names=['T_F', 'file_score', 'label', 'all_text'],
                           keep_default_na=False)
    test_df = pd.read_csv(tst_fn, sep='\t', header=None, names=['T_F', 'file_score', 'label', 'all_text'],
                          keep_default_na=False)

    train_df['file'] = train_df['file_score'].apply(lambda x: x.split('#')[0])
    train_df['score'] = train_df['file_score'].apply(lambda x: x.split('#')[1])
    test_df['file'] = test_df['file_score'].apply(lambda x: x.split('#')[0])
    test_df['score'] = test_df['file_score'].apply(lambda x: x.split('#')[1])

    print("Size of train df: ", train_df.shape[0])
    print("Size of test df: ", test_df.shape[0])
    # remove false labels for file_score also occurring with true label
    true_ids = train_df[train_df['T_F'] == True]['file_score'].tolist()
    # tmp_train_df.drop(tmp_train_df[tmp_train_df['T_F'] == False].index, inplace=True)
    # tmp_test_df.drop(tmp_test_df[tmp_test_df['T_F'] == False].index, inplace=True)
    # print(train_df[:10])

    # loop over rows and remove row with false and file_score matching true label
    # TODO vectorize
    for index, row in train_df.iterrows():
        if row['T_F'] == False and row['file_score'] in true_ids:
            train_df.drop(index, inplace=True)

    print("Size of train df: ", train_df.shape[0])
    print("Size of test df: ", test_df.shape[0])

    return train_df, test_df


### Preprocessing ###

# load train/test splits
train_fn = "../../data/exp1/train_score.tsv"
test_fn = "../../data/exp1/test_score.tsv"

train_df, test_df = load_score_data(train_fn, test_fn)

# print stats on train and test sets?
print("Train:")
print("Files: ", train_df['file_score'].nunique())
print("Labels: ", len(set([t for sublist in train_df['label'].tolist() for t in sublist])))
print("Test:")
print("Files: ", test_df['file_score'].nunique())
print("Labels: ", len(set([t for sublist in test_df['label'].tolist() for t in sublist])))


# apply tfidfvectorizer to get features
vectorizer = TfidfVectorizer(sublinear_tf=False, max_df=0.95)
# fit on train
vectorizer = vectorizer.fit(train_df['all_text'].values)
x_train = vectorizer.transform(train_df['all_text'].values)
x_test = vectorizer.transform(test_df['all_text'].values)
print("Number of features: ", x_train.shape[1])
print("Number of (train) samples: ", x_train.shape[0])

# get instance labels: multilabel or not?
# task, data, and metrics label
le = LabelBinarizer().fit(train_df['T_F'].tolist())
y = le.transform(train_df['T_F'].tolist())
y_test = le.transform(test_df['T_F'].tolist())

# print shapes, sizes
print("X:", x_train.shape, x_test.shape)
print("y:", y.shape, y_test.shape)

### Training ###
# random shuffling might not be ideal for splitting this data, but I don't think we have another option

clf = LogisticRegression().fit(x_train, y)
print("Trained model.")

### Prediction ###
# run prediction on a couple of examples from the test data
print("Raw text: ", test_df['all_text'][0])
print("Some features: ")
print("Gold label: ", test_df['T_F'][0], " (binary) ", y_test[0])
print("Prediction: ", clf.predict(x_test[0]))
print("Prediction prob.: ", clf.predict_proba(x_test[0]))


# run prediction, one file at a time.  return best score for file
pdf_names = test_df['file'].unique()

scores = []
for pdf_name in pdf_names:
    pdf_df = test_df[test_df['file'] == pdf_name].reset_index()
    x_pdf = vectorizer.transform(pdf_df['all_text'].values)
    pred_prob = clf.predict_proba(x_pdf)
    # get index of highest true prediction
    am = np.argmax(pred_prob[:,1])  # need max of second column, prob label is true
    # print(am, pdf_df['score'][am])
    scores.append(pdf_df['score'][am])


### Print results ###
# dump to CSV for evaluation
# want to write paper name, task, dataset, evaluation metric, score
csv_df = pd.DataFrame(data={'file': pdf_names, 'score': scores})
out_fn = "/tmp/score_output.csv"
print("Printing score output to ", out_fn)
csv_df.to_csv(out_fn, index=False)
