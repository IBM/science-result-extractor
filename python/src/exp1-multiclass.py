# coding: utf-8
# # Create NLP Leaderboards #
# 
# We want to take NLP papers and assign them to a leaderboard by task (T), dataset (D), and evaluation metric (M).

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix

LAMBDA = 0.01


def load_exp1_data(trn_fn, tst_fn):
    tmp_train_df = pd.read_csv(trn_fn, sep='\t', header=None, names=['T_F', 'file', 'label', 'all_text'])
    tmp_test_df = pd.read_csv(tst_fn, sep='\t', header=None, names=['T_F', 'file', 'label', 'all_text'])

    print("Size of train df: ", tmp_train_df.shape[0])
    print("Size of test df: ", tmp_test_df.shape[0])
    # remove false labels, not needed for multiclass classification
    tmp_train_df.drop(tmp_train_df[tmp_train_df['T_F'] == False].index, inplace=True)
    tmp_test_df.drop(tmp_test_df[tmp_test_df['T_F'] == False].index, inplace=True)
    print("Size of train df: ", tmp_train_df.shape[0])
    print("Size of test df: ", tmp_test_df.shape[0])
    # print(train_df[:10])

    # loop over rows and merge labels for same instances
    unk_limit = 3
    unk_seen = 0
    train_instance_d = {}
    for index, row in tmp_train_df.iterrows():
        if row['label'] == 'unknow':
            if unk_seen < unk_limit:
                unk_seen += 1
            else:
                continue  # don't train on unknown labels
        if row['file'] in train_instance_d:
            if row['all_text'] in train_instance_d[row['file']]:
                train_instance_d[row['file']][row['all_text']].append(row['label'])
            else:
                assert False, "All files should match one and only one text."
        else:
            train_instance_d[row['file']] = {row['all_text']: [row['label']]}

    # loop over rows and merge labels for same instances
    test_instance_d = {}
    for index, row in tmp_test_df.iterrows():
        if row['file'] in test_instance_d:
            if row['all_text'] in test_instance_d[row['file']]:
                test_instance_d[row['file']][row['all_text']].append(row['label'])
            else:
                assert False, "All files should match one and only one text."
        else:
            test_instance_d[row['file']] = {row['all_text']: [row['label']]}

    print("Train files: ", len(train_instance_d))
    print("Test files: ", len(test_instance_d))
    # print("Train instances: ", sum([len(v.items()) for k,v in train_instance_d.items()]))
    # print("Test instances: ", sum([len(v.items()) for k,v in test_instance_d.items()]))
    # print(train_instance_d['trouillon16.pdf'][
    #           'Complex Embeddings for Simple Link Prediction In statistical relational learning, the link prediction problem is key to automatically understand the structure of large knowledge bases. As in previous studies, we propose to solve this problem through latent factorization. However, here we make use of complex valued embeddings. The composition of complex embeddings can handle a large variety of binary relations, among them symmetric and antisymmetric relations. Compared to state-of-the-art models such as Neural Tensor Network and Holographic Embeddings, our approach based on complex embeddings is arguably simpler, as it only uses the Hermitian dot product, the complex counterpart of the standard dot product between real vectors. Our approach is scalable to large datasets as it remains linear in both space and time, while consistently outperforming alternative approaches on standard link prediction benchmarks. 1 In order to evaluate our proposal, we conducted experiments on both synthetic and real datasets The synthetic dataset is based on relations that are either symmetric or antisymmetric, whereas the real datasets comprise different types of relations found in different, standard KBs Dataset We next evaluate the performance of our model on the FB15K and WN18 datasets summarizes the metadata of the two datasets Both datasets contain only positive triples For evaluation, we measure the quality of the ranking of each test triple among all possible subject and object substitutions : r(s , o) and r(s, o ), ∀s , ∀o ∈ E Mean Reciprocal Rank (MRR) and Hits at mare the standard evaluation measures for these datasets and come in two flavours: raw and filtered) We report both filtered and raw MRR, and filtered Hits at 1, 3 and 10 in for the evaluated models Furthermore, we chose TransE, DistMult Table 3. Number of entities, relations, and observed triples in each split for the FB15K and WN18 datasets. |R| Table 2. Filtered and Raw Mean Reciprocal Rank (MRR) for the models tested on the FB15K and WN18 datasets. Hits@m metrics are filtered. *Results reported from (Nickel et al., 2016b) for HolE model. 1 FB15K 3 Filter WN18 Hits at Raw MRR 10 Table 4. Filtered Mean Reciprocal Rank (MRR) for the models tested on each relation of the Wordnet dataset (WN18). ComplEx DistMult TransE'])

    # convert to dictionaries for creating new dataframe
    train_d = {'file': [], 'all_text': [], 'task': [], 'dataset': [], 'metric': []}
    test_d = {'file': [], 'all_text': [], 'task': [], 'dataset': [], 'metric': []}

    for f, t_d in train_instance_d.items():
        assert len(t_d) == 1, "Should only be one entry in this dict."
        for t, labels in t_d.items():  # should only be one
            train_d['file'].append(f)
            train_d['all_text'].append(t)
            tasks, datasets, metrics = parse_labels(labels, is_train=True)
            train_d['task'].append(tasks)
            train_d['dataset'].append(datasets)
            train_d['metric'].append(metrics)

    for f, t_d in test_instance_d.items():
        assert len(t_d) == 1, "Should only be one entry in this dict."
        for t, labels in t_d.items():  # should only be one
            test_d['file'].append(f)
            test_d['all_text'].append(t)
            tasks, datasets, metrics = parse_labels(labels)
            test_d['task'].append(tasks)
            test_d['dataset'].append(datasets)
            test_d['metric'].append(metrics)

    return pd.DataFrame(data=train_d), pd.DataFrame(data=test_d)


def parse_labels(labels_, is_train=False):
    tasks_ = set()
    datasets_ = set()
    metrics_ = set()
    for label_ in labels_:
        if label_ == "unknow":
            task, dataset, metric = label_, label_, label_
        else:
            try:
                task, dataset, metric = [x.strip() for x in label_.split(',')]
                if is_train:
                    increment_label_counts(task, dataset, metric)
            except ValueError:
                print("Unable to parse label: ", label_)
        tasks_.add(task)
        datasets_.add(dataset)
        metrics_.add(metric)
    return tasks_, datasets_, metrics_


def increment_label_counts(tsk, ds, met):
    global trigram_d  # order metric, data, task
    global bigram_d  # order metric, data  and metric, task  and data, task
    global unigram_d
    if met in trigram_d:
        if ds in trigram_d[met]:
            if tsk in trigram_d[met][ds]:
                trigram_d[met][ds][tsk] += 1
            else:
                trigram_d[met][ds][tsk] = 1
        else:
            trigram_d[met][ds] = {tsk: 1}
    else:
        trigram_d[met] = {ds: {tsk: 1}}
    # if met in bigram_d:
    #     if ds in bigram_d[met]:
    #         bigram_d[met][ds] += 1
    #     else:
    #         bigram_d[met][ds] = 1
    #     if tsk in bigram_d[met]:
    #         bigram_d[met][tsk] += 1
    #     else:
    #         bigram_d[met][tsk] = 1
    # else:
    #     bigram_d[met] = {ds: 1, tsk: 1}
    if ds in bigram_d:
        if tsk in bigram_d[ds]:
            bigram_d[ds][tsk] += 1
        else:
            bigram_d[ds][tsk] = 1
    else:
        bigram_d[ds] = {tsk: 1}
    for l in [met, ds, tsk]:
        if l in unigram_d:
            unigram_d[l] += 1
        else:
            unigram_d[l] = 1


def mle(trigram_d, bigram_d, unigram_d, lab_enc):
    mle = {}
    for m in lab_enc['metric'].classes_:
        if m == 'unknow':
            continue
        mle[m] = {}
        for d in lab_enc['dataset'].classes_:
            if d == 'unknow':
                continue
            mle[m][d] = {}
            for t in lab_enc['task'].classes_:
                if t == 'unknow':
                    continue
                num = LAMBDA
                denom = len(lab_enc['metric'].classes_)*LAMBDA
                if d in bigram_d and t in bigram_d[d]:
                    denom += bigram_d[d][t]
                    if m in trigram_d and d in trigram_d[m] and t in trigram_d[m][d]:
                        num += trigram_d[m][d][t]
                mle[m][d][t] = num/denom

    for d in lab_enc['dataset'].classes_:
        if d == 'unknow':
            continue
        mle[d] = {}
        for t in lab_enc['task'].classes_:
            if t == 'unknow':
                continue
            num = LAMBDA
            denom = len(lab_enc['dataset'].classes_)*LAMBDA
            if t in unigram_d:
                denom += unigram_d[t]
                if d in bigram_d and t in bigram_d[d]:
                    num += bigram_d[d][t]
            mle[d][t] = num/denom

    return mle


def update_pred(pred_, pred_prob_):
    update_pred_ = {}
    for label_ in ['task', 'dataset', 'metric']:
        pred_prob_np = None
        for i in pred_prob_[label_]:  # use for-loop to make sure all have the right shape
            if i.shape[1] == 1:  # only has one column, we need to add another
                i = np.concatenate((i, np.zeros((i.shape[0], 1))), axis=1)
            assert i.shape[1] == 2, "Only expecting 2 columns (at this point) but we have shape: %s" % str(i.shape)
            if pred_prob_np is None:
                pred_prob_np = i
            else:
                pred_prob_np = np.concatenate((pred_prob_np, i))
        pred_prob_np = pred_prob_np.reshape((len(pred_prob_[label_]), -1, 2))
        am = np.argmax(pred_prob_np, axis=0)[:, 1]  # need max of second column, prob label is true
        oh = np.zeros(pred_[label_].shape)
        oh[np.arange(pred_[label_].shape[0]), am] = 1
        print(((oh + pred_[label_]) > 0)[:5])
        update_pred_[label_] = (oh + pred_[label_]) > 0  # combine original prediction and constrainted prediction
    return update_pred_


def conditional_prob_metric(lab_enc, mle):
    # get t classes
    t_names = lab_enc['task'].classes_
    # get d classes
    d_names = lab_enc['dataset'].classes_
    # get m classes
    m_names = lab_enc['metric'].classes_
    num_ts = len(t_names)
    num_ds = len(d_names)
    num_ms = len(m_names)
    m_dt_ls = []
    for m in m_names:
        if m == 'unknow':
            continue
        for d in d_names:
            if d == 'unknow':
                continue
            for t in t_names:
                if t == 'unknow':
                    continue
                m_dt_ls.append(mle[m][d][t])  # should all be in this dictionary
    assert len(m_dt_ls) == (num_ts-1) * (num_ds-1) * (num_ms-1), \
        "This list should be a long as product of 3 lists' lengths (excluding 'unknow')"
    return np.array(m_dt_ls).reshape(len(m_names)-1, len(d_names)-1, len(t_names)-1)


def conditional_prob_dataset(lab_enc, mle):
    # get t classes
    t_names = lab_enc['task'].classes_
    # get d classes
    d_names = lab_enc['dataset'].classes_
    num_ts = len(t_names)
    num_ds = len(d_names)
    d_t_ls = []
    for d in d_names:
        if d == 'unknow':
            continue
        for t in t_names:
            if t == 'unknow':
                continue
            d_t_ls.append(mle[d][t])  # should all be in this dictionary
    assert len(d_t_ls) == (num_ts-1) * (num_ds-1), \
        "This list should be a long as product of 2 lists' lengths (excluding the unknown class)"
    return np.array(d_t_ls).reshape(len(d_names)-1, len(t_names)-1)


def chain_rule(pred_, pred_prob_, lab_enc, mle):
    # get t classes
    # t_names = lab_enc['task'].classes_
    # get d classes
    d_cnt = len(lab_enc['dataset'].classes_) - 1  # ignore unknown
    # get m classes
    m_cnt = len(lab_enc['metric'].classes_) - 1  # ignore unknown

    p_m_td = conditional_prob_metric(lab_enc, mle)
    p_d_t = conditional_prob_dataset(lab_enc, mle)

    # skip 'unknown' prediction

    tsk_pred_prob = np.array([tsk for tsk_i, tsk in enumerate(pred_prob_['task']) if tsk_i != unk_index['task']])
    # get only prob of task
    update_lab = row_chain_rule(tsk_pred_prob[:,:,1], p_m_td, p_d_t, m_cnt, d_cnt)

    update_pred_ = {}
    for lab in ['task', 'dataset', 'metric']:
        update_pred_[lab] = np.zeros((update_lab[lab].shape[0], len(lab_enc[lab].classes_) - 1))
        update_pred_[lab][np.arange(update_pred_[lab].shape[0]), update_lab[lab]] = 1
        # add empty column for removed 'uknown'
        update_pred_[lab] = np.insert(update_pred_[lab], unk_index[lab], 0, axis=1)
    return update_pred_


def row_chain_rule(task_pred, p_m_td, p_d_t, num_m, num_d):
    # we're going to lose multi-label option here
    tile_pred = np.tile(task_pred, (num_m * num_d, 1))
    conds = (p_m_td * p_d_t).reshape(p_m_td.size, -1)
    m = (conds * tile_pred).reshape((num_m, num_d, task_pred.shape[0], task_pred.shape[1]))
    m = m.reshape((num_m*num_d*task_pred.shape[0], task_pred.shape[1]))
    max_ = np.argmax(m, axis=0)
    ind = np.unravel_index(np.argmax(m, axis=0), (num_m, num_d, task_pred.shape[0]))
    return {'metric': ind[0], 'dataset': ind[1], 'task': ind[2]}


### Preprocessing ###

# load train/test splits
# train_fn = "../../data/exp1/train_positive.tsv"
# test_fn = "../../data/exp1/test_positive.tsv"
train_fn = "../../data/exp1/ablationfull/train.tsv"
test_fn = "../../data/exp1/ablationfull/test.tsv"

# counts for pairs and triples for computing MLE
trigram_d = {}  # order metric, data, task
bigram_d = {}  # order metric, data  and metric, task  and data, task
unigram_d = {}

train_df, test_df = load_exp1_data(train_fn, test_fn)

# print stats on train and test sets?
print("Train:")
print("Files: ", train_df['file'].nunique())
print("Tasks: ", len(set([t for sublist in train_df['task'].tolist() for t in sublist])))
print("Datasets: ", len(set([t for sublist in train_df['dataset'].tolist() for t in sublist])))
print("Metrics: ", len(set([t for sublist in train_df['metric'].tolist() for t in sublist])))
print("Test:")
print("Files: ", test_df['file'].nunique())
print("Tasks: ", len(set([t for sublist in test_df['task'].tolist() for t in sublist])))
print("Datasets: ", len(set([t for sublist in test_df['dataset'].tolist() for t in sublist])))
print("Metrics: ", len(set([t for sublist in test_df['metric'].tolist() for t in sublist])))


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
le = {}  # dict for label encoders
y = {}  # dict for labels
y_test = {}
unk_index = {}
for label in ['task', 'dataset', 'metric']:
    # multilabel
    le[label] = MultiLabelBinarizer().fit(train_df[label].tolist() + test_df[label].tolist())
    y[label] = le[label].transform(train_df[label].tolist())
    y_test[label] = le[label].transform(test_df[label].tolist())
    unk_index[label] = np.nonzero(le[label].classes_ == 'unknow')[0][0]

# print shapes, sizes
print("X:", x_train.shape, x_test.shape)
print("y:", y['task'].shape, y_test['task'].shape)
for label in ['task', 'dataset', 'metric']:
    print(le[label].classes_)
    print(unk_index[label])

### Training ###
# random shuffling might not be ideal for splitting this data, but I don't think we have another option

# simple_clf = RandomForestClassifier()
simple_clf = LogisticRegression(solver="saga", multi_class="multinomial")

clf = {}  # dict for classifier (by label)
for label in ['task', 'dataset', 'metric']:
    clf[label] = MultiOutputClassifier(simple_clf, n_jobs=-1).fit(x_train, y[label])
#     clf[label] = RandomForestClassifier(n_jobs=-1).fit(x_train, y[label])
    print("Trained model for ", label)

### Prediction ###
# run prediction on a couple of examples from the test data
print("Raw text: ", test_df['all_text'][0])
print("Some features: ")
for label in ['task', 'dataset', 'metric']:
    print("Gold label: ", test_df[label][0], " (binary) ", y_test[label][0])
    print("Prediction: ", clf[label].predict(x_test[0]))
    print("Prediction prob.: ", clf[label].predict_proba(x_test[0]))

# run prediction on all test data
pred = {}
pred_prob = {}
for label in ['task', 'dataset', 'metric']:
    pred[label] = clf[label].predict(x_test)
    pred_prob[label] = clf[label].predict_proba(x_test)
    score = f1_score(y_test[label], pred[label], average='macro')
    print(score)

mle_d = mle(trigram_d, bigram_d, unigram_d, le)

# apply some constraints on predictions
# treat labels independently, i.e., no joint modeling
# update_pred = update_pred(pred, pred_prob)
# first joint attempt model P(m,d,t) as P(m|d,t) * P(d|t) * P(t) with chain rule
update_pred = chain_rule(pred, pred_prob, le, mle_d)

### Print results ###
# dump to CSV for evaluation
# want to write paper name, task, dataset, evaluation metric, score
csv_df = test_df['file'].copy()
for label in ['task', 'dataset', 'metric']:
    csv_df = pd.concat((csv_df, pd.Series(le[label].inverse_transform(update_pred[label]))), axis=1)
csv_df['score'] = '0.0'
csv_df.columns = ['file', 'task', 'dataset', 'metric', 'score']
out_fn = "/tmp/output-splitlabel.csv"
print("Printing to ", out_fn)
csv_df.to_csv(out_fn, index=False)
