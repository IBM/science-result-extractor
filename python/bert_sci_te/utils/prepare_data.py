import numpy as np
import pandas as pd
from collections import Counter
import tensorflow as tf
import re
from keras.preprocessing.sequence import pad_sequences
# names = ["class", "title", "content"]
names = ["Quality", "#1 ID", "#1 ID", "#1 String", "#2 String"]
from nltk.tokenize import word_tokenize
import re
import collections
import pickle


def tokenize_bert(sent):
    # return [x.strip() for x in re.split('(\s+)?', sent) if x.strip()]
    return [x.strip().lower() for x in re.split('(\s+)?', sent) if x.strip()]

def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"@[A-Za-z0-9]+", " ", text)
    text = text.strip().lower()
    return text

def to_one_hot(y, n_class):
    return np.eye(n_class)[y.astype(int)]


def load_data(file_name, sample_ratio=1, n_class=15, names=names, one_hot=True):
    '''load data from .csv file'''
    csv_file = pd.read_csv(file_name, names=names)
    shuffle_csv = csv_file.sample(frac=sample_ratio)
    x = pd.Series(shuffle_csv["content"])
    y = pd.Series(shuffle_csv["class"])
    if one_hot:
        y = to_one_hot(y, n_class)
    return x, y

def build_dataset_is(tsv, word_dict, document_max_len, names=names):
    csv_file = pd.read_csv(tsv, names=names, sep='\t', header=0)
    df = pd.Series(csv_file["#2 String"])
    y1 = []
    iscode = {"old_np":1,
              "new_np":2,
              "mediated_bridging_np":3,
              "mediated_syntactic_np":4,
              "mediated_comparative_np":5,
              "mediated_aggregate_np":6,
              "mediated_func_np":7,
              "mediated_world_knowledge_or_text_np":8}
    # x = list(map(lambda d: word_tokenize(clean_str(d)), df))
    x = list(map(lambda d: tokenize_bert(d), df))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<padding>"]], x))
    y = pd.Series(csv_file["Quality"])
    for label in y:
        y1.append(iscode.get(label))
    return x, y1

def load_data_is(file_name, vocab, n_class=8, names=names, one_hot=True):
    '''load data from .csv file'''
    x1, y1 = [],[]
    iscode = {"old_np":1,
              "new_np":2,
              "mediated_bridging_np":3,
              "mediated_syntactic_np":4,
              "mediated_comparative_np":5,
              "mediated_aggregate_np":6,
              "mediated_func_np":7,
              "mediated_world_knowledge_or_text_np":8}
    csv_file = pd.read_csv(file_name, names=names, sep='\t', header=0)
    # shuffle_csv = csv_file.sample(frac=sample_ratio)
    x = pd.Series(csv_file["#2 String"])
    for sent in x:
        anasent_idx = map_to_idx(tokenize(sent), vocab)
        x1.append(anasent_idx)
    y = pd.Series(csv_file["Quality"])
    for label in y:
        y1.append(iscode.get(label))


    if one_hot:
        y = to_one_hot(y, n_class)
    return x1, y1


def build_dict(full_tsv, is_train=True):
    if is_train:
        df = pd.read_csv(full_tsv, sep="\t")
        sentences = df["#2 String"]
        words = list()
        for sentence in sentences:
            # for word in word_tokenize(clean_str(sentence)):
            for word in tokenize_bert(sentence):
                words.append(word)
        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)
    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))
    return word_dict, reversed_dict


def data_preprocessing(train, test, max_len):
    """transform to one-hot idx vector by VocabularyProcessor"""
    """VocabularyProcessor is deprecated, use v2 instead"""
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len)
    x_transform_train = vocab_processor.fit_transform(train)
    x_transform_test = vocab_processor.transform(test)
    vocab = vocab_processor.vocabulary_
    vocab_size = len(vocab)
    x_train_list = list(x_transform_train)
    x_test_list = list(x_transform_test)
    x_train = np.array(x_train_list)
    x_test = np.array(x_test_list)

    return x_train, x_test, vocab, vocab_size

def tokenize(sent):
    '''
    data_reader.tokenize('a#b')
    ['a', '#', 'b']
    '''
    # return [x.strip().lower() for x in re.split('(\W+)?', sent) if x.strip()]
    # return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
    return [x.strip().lower() for x in re.split('(\s+)?', sent) if x.strip()]
    # return [x.strip() for x in re.split('(\s+)?', sent) if x.strip()]

def get_vocab(data):
    vocab=Counter()
    for ex in data:
        tokens=tokenize(ex.lower())
        vocab.update(tokens)
        # print vocab
        # sys.exit()
    lst = ["unk", "delimiter"] + [ x for x, y in vocab.items() if y > 0]
    vocab = dict([ (y,x) for x,y in enumerate(lst) ])
    return vocab


def map_to_txt(x, vocab):
    textify = map_to_idx(x, inverse_map(vocab))
    return ' '.join(textify)


def inverse_map(vocab):
    return {v: k for k, v in vocab.items()}

def map_to_idx(x, vocab):
    '''
    x is a sequence of tokens
    '''
    # 0 is for UNK
    return [ vocab[w] if w in vocab else 0 for w in x  ]

def data_preprocessing_v2(train, test, max_len, max_words=50000):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train)
    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, test_padded, max_words + 2


def data_preprocessing_with_dict(train, test, max_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
    tokenizer.fit_on_texts(train)
    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, test_padded, tokenizer.word_docs, tokenizer.word_index, len(tokenizer.word_docs) + 2


def split_dataset(x_test, y_test, dev_ratio):
    """split test dataset to test and dev set with ratio """
    test_size = len(x_test)
    print(test_size)
    dev_size = (int)(test_size * dev_ratio)
    print(dev_size)
    x_dev = x_test[:dev_size]
    x_test = x_test[dev_size:]
    y_dev = y_test[:dev_size]
    y_test = y_test[dev_size:]
    return x_test, x_dev, y_test, y_dev, dev_size, test_size - dev_size


def fill_feed_dict(data_X, data_Y, batch_size):
    """Generator to yield batches"""
    # Shuffle data first.
    # shuffled_X, shuffled_Y = shuffle(data_X, data_Y)
    # shuffled_X, shuffled_Y = data_X, data_Y
    # print("before shuffle: ", data_Y[:10])
    # print(data_X.shape[0])
    # perm = np.random.permutation(data_X.shape[0])
    # data_X = data_X[perm]
    # shuffled_Y = data_Y[perm]
    # print("after shuffle: ", shuffled_Y[:10])
    # for idx in range(len(data_X)-1 // batch_size+1):
    #     x_batch = shuffled_X[batch_size * idx: batch_size * (idx + 1)]
    #     y_batch = shuffled_Y[batch_size * idx: batch_size * (idx + 1)]
    #     yield x_batch, y_batch

    num_batches_per_epoch = (len(data_X) - 1) // batch_size + 1

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(data_X))
        yield data_X[start_index:end_index], data_Y[start_index:end_index]


