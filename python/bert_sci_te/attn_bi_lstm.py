from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import BasicLSTMCell
from utils.prepare_data import *
import time
from utils.model_helper import *
import tensorflow as tf
import sys
import logging
from keras.callbacks import *
#from visualizer import *
from keras.models import *
from keras.optimizers import *



from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np


def get_init_embedding(reversed_dict, glovefile):
    print("Loading Glove vectors...")
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glovefile, word2vec_file)
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)
    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros([config.get("embedding_size")], dtype=np.float32)
        word_vec_list.append(word_vec)
    return np.array(word_vec_list)

def layer_normalization(inputs,
                        epsilon=1e-8,
                        scope="ln",
                        reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)
        outputs = gamma * normalized + beta

    return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=12,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:  # set default size for attention size C
            num_units = queries.get_shape().as_list()[-1]

        # Linear Projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # [N, T_q, C]
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # [N, T_k, C]
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # [N, T_k, C]

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)  # [num_heads * N, T_q, C/num_heads]
        K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)  # [num_heads * N, T_k, C/num_heads]
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)  # [num_heads * N, T_k, C/num_heads]

        # Attention
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (num_heads * N, T_q, T_k)

        # Scale : outputs = outputs / sqrt( d_k)
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        # see : https://github.com/Kyubyong/transformer/issues/3
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # -infinity
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation: outputs is a weight matrix
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # reshape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # residual connection
        outputs += queries

        # layer normaliztion
        outputs = layer_normalization(outputs)
        return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        print("Conv ret:", outputs.shape)
        # Residual connection
        outputs += inputs

        # Normalize
        outputs = layer_normalization(outputs)

    return outputs


class AttentionClassifier(object):
    def __init__(self, config, reversed_dict):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = len(reversed_dict)
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]

        # placeholder
        self.x = tf.placeholder(tf.int32, [None, self.max_len])
        self.label = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.glovefile = config["glove"]

        init_embeddings = tf.constant(get_init_embedding(reversed_dict, self.glovefile), dtype=tf.float32)

        self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=False)


    def build_graph(self):
        print("building graph...")
        # embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
        #                              trainable=True)
        # batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.x)
        batch_embedded = tf.nn.embedding_lookup(self.embeddings, self.x)

        # multi-head attention
        ma = multihead_attention(queries=batch_embedded, keys=batch_embedded)
        # FFN(x) = LN(x + point-wisely NN(x))
        outputs = feedforward(ma, [self.hidden_size, self.embedding_size])
        outputs = tf.reshape(outputs, [-1, self.max_len * self.embedding_size])
        logits = tf.layers.dense(outputs, units=self.n_class)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.label))
        self.prediction = tf.argmax(tf.nn.softmax(logits), 1)

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        print("graph built successfully!")


class ABLSTM(object):
    def __init__(self, config, reversed_dict):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = len(reversed_dict)
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]

        # placeholder
        self.x = tf.placeholder(tf.int32, [None, self.max_len])
        self.label = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.glovefile = config["glove"]

        init_embeddings = tf.constant(get_init_embedding(reversed_dict, self.glovefile), dtype=tf.float32)

        self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=False)






    def build_graph(self):
        print("building graph")
        # Word embedding
        # embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
        #                              trainable=False)
        batch_embedded = tf.nn.embedding_lookup(self.embeddings, self.x)
        rnn_outputs, _ = bi_rnn(BasicLSTMCell(self.hidden_size),
                                BasicLSTMCell(self.hidden_size),
                                inputs=batch_embedded, dtype=tf.float32)

        fw_outputs, bw_outputs = rnn_outputs

        W = tf.Variable(tf.random_normal([self.hidden_size], stddev=0.1))
        H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
        M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)

        self.alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, self.hidden_size]),
                                                        tf.reshape(W, [-1, 1])),
                                              (-1, self.max_len)))  # batch_size x seq_len
        r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                      tf.reshape(self.alpha, [-1, self.max_len, 1]))
        r = tf.squeeze(r)
        h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE

        h_drop = tf.nn.dropout(h_star, self.keep_prob)

        # Fully connected layer（dense layer)
        FC_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_class], stddev=0.1))
        FC_b = tf.Variable(tf.constant(0., shape=[self.n_class]))
        y_hat = tf.nn.xw_plus_b(h_drop, FC_W, FC_b)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=self.label))

        # prediction
        self.prediction = tf.argmax(tf.nn.softmax(y_hat), 1)

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        print("graph built successfully!")


class BLSTM(object):
    def __init__(self, config, reversed_dict):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = len(reversed_dict)
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]

        # placeholder
        self.x = tf.placeholder(tf.int32, [None, self.max_len])
        self.label = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.glovefile = config["glove"]

        init_embeddings = tf.constant(get_init_embedding(reversed_dict, self.glovefile), dtype=tf.float32)

        self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=False)


    def build_graph(self):
        print("building graph")
        # Word embedding
        # embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
        #                              trainable=False)
        batch_embedded = tf.nn.embedding_lookup(self.embeddings, self.x)
        rnn_outputs, _ = bi_rnn(BasicLSTMCell(self.hidden_size),
                                BasicLSTMCell(self.hidden_size),
                                inputs=batch_embedded, dtype=tf.float32)

        fw_outputs, bw_outputs = rnn_outputs

        W = tf.Variable(tf.random_normal([self.hidden_size], stddev=0.1))
        H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
        M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)

        self.alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, self.hidden_size]),
                                                        tf.reshape(W, [-1, 1])),
                                              (-1, self.max_len)))  # batch_size x seq_len
        r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                      tf.reshape(self.alpha, [-1, self.max_len, 1]))
        r = tf.squeeze(r)
        h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE

        h_star = H[:,-1]
        h_drop = tf.nn.dropout(h_star, self.keep_prob)

        # Fully connected layer（dense layer)
        FC_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_class], stddev=0.1))
        FC_b = tf.Variable(tf.constant(0., shape=[self.n_class]))
        y_hat = tf.nn.xw_plus_b(h_drop, FC_W, FC_b)


        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=self.label))

        # prediction
        self.prediction = tf.argmax(tf.nn.softmax(y_hat), 1)

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        print("graph built successfully!")




if __name__ == '__main__':
    config = {
        "max_len": 32,
        "hidden_size": 768,
        "vocab_size": 30000,
        "embedding_size": 768,
        "n_class": 9,
        "learning_rate": 5e-5,
        # "learning_rate": 1e-3,
        "batch_size": 64,
        "train_epoch": 20,

         "train":"data/is_new_prediction2feat1/fold0/train_bertemb.tsv",
         "test":"data/is_new_prediction2feat1/fold0/test_bertemb.tsv",
         # "glove": "/u/yufang/data/glove/glove.6B.100d.txt"
         "glove":"data/is_new_prediction2feat1/fold0/fullstr_bertemb.txt",
         "full":"data/is_new_prediction2feat1/fold0/full_bertemb.tsv",
         "prediction":"data/is_new_prediction2feat1/fold0/prediction"

       # "train":"data/bridgingAna/is_new_prediction2discoursefeat1/fold0/train.tsv",
       # "test":"data/bridgingAna/is_new_prediction2discoursefeat1/fold0/test.tsv",
       # "glove": "/Users/yhou/git/bert/data/glove.6B/glove.6B.300d.txt",
       # "full":"data/bridgingAna/is_new_prediction2discoursefeat1/fold0/full.tsv",
        #"prediction":"/Users/yhou/git/bert/data/bridgingAna/is_new_prediction2discoursefeat1/fold0/prediction.tsv"

        # "train":"data/bridgingAna/is_new_prediction2discoursefeat1/fold0/train_bertemb.tsv",
        # "test": "data/bridgingAna/is_new_prediction2discoursefeat1/fold0/test_bertemb.tsv",
        # "glove": "data/bridgingAna/is_new_prediction2discoursefeat1/fold0/fullstr_bertemb.txt",
        # "full": "data/bridgingAna/is_new_prediction2discoursefeat1/fold0/full_bertemb.tsv",
        # "prediction": "/Users/yhou/git/bert/data/bridgingAna/is_new_prediction2discoursefeat1/fold0/prediction.tsv"

    }

    word_dict, reversed_dict = build_dict(config.get("full"))
    print("Building dataset...")
    x_train, y_train = build_dataset_is(config.get("train"), word_dict, config.get("max_len"))
    # print(x_train[:2])
    # sys.exit()
    x_test, y_test = build_dataset_is(config.get("test"), word_dict, config.get("max_len"))




    sess = tf.Session()
    #classifier  = AttentionClassifier(config, reversed_dict)
    # classifier = ABLSTM(config, reversed_dict)
    classifier = BLSTM(config, reversed_dict)
    classifier.build_graph()
    sess.run(tf.global_variables_initializer())
    start = time.time()
    for e in range(config["train_epoch"]):

        t0 = time.time()
        print("Epoch %d start !" % (e + 1))
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, config["batch_size"]):
            return_dict = run_train_step(classifier, sess, (x_batch, y_batch))
            # attn = get_attn_weight(classifier, sess, (x_batch, y_batch))
            # plot the attention weight
            # print(np.reshape(attn, (config["batch_size"], config["max_len"])))
        t1 = time.time()

        print("Train Epoch time:  %.3f s" % (t1 - t0))
        dev_cnt = 0
        dev_acc = 0
        for x_batch, y_batch in fill_feed_dict(x_test, y_test, config["batch_size"]):
            acc, batch_prediction = run_eval_step(classifier, sess, (x_batch, y_batch))
            dev_acc += acc
            dev_cnt += 1

        print("Dev accuracy : %f %%" % (dev_acc / dev_cnt * 100))

    print("Training finished, time consumed : ", time.time() - start, " s")
    print("Start evaluating:  \n")
    cnt = 0
    test_acc = 0
    prediction = []
    for x_batch, y_batch in fill_feed_dict(x_test, y_test, config["batch_size"]):
        acc, batch_prediction = run_eval_step(classifier, sess, (x_batch, y_batch))
        prediction.append(batch_prediction)
        test_acc += acc
        cnt += 1

    print("Test accuracy : %f %%" % (test_acc / cnt * 100))
    with open(config.get("prediction"),"w") as f:
        for predict in prediction:
            for p in predict:
                f.write(str(p) + "\n")
