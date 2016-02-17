import logging
import math
import random
import numpy as np
import joblib
import numpy as np
import os
import time
import datetime
import tensorflow as tf
from gensim import models
import feature_extraction


logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
gensim_model = None

class DeepCNNAqquRelScorer():

    UNK = '---UNK---'

    def __init__(self, name, embedding_file):
        self.name = name + "_DeepRelScorer"
        [self.embedding_size, self.vocab,
         self.embeddings] = self.extract_vectors(embedding_file)
        self.UNK_ID = self.vocab[DeepCNNAqquRelScorer.UNK]
        # This is the maximum number of tokens in a query we consider.
        self.max_query_len = 40
        self.filter_sizes = [2, 3, 4]
        self.sentence_len = self.max_query_len + 2 * (max(self.filter_sizes) - 1)
        self.rel_width_len = 6 * self.embedding_size

    def extract_vectors(self, gensim_model_fname):
        """Extract vectors from gensim model and add UNK/PAD vectors.
        """
        logger.info("Preparing embeddings matrix")
        global gensim_model
        if gensim_model is None:
            gensim_model = models.Word2Vec.load(gensim_model_fname)
        vector_size = gensim_model.vector_size
        common_words = set()
        # Find the most frequent words to keep the embeddings small.
        # TF doesn't work when they are larger than 2GB !?
        with open("data/word_frequencies.txt") as f:
            for line in f:
                cols = line.strip().split()
                if len(cols) != 2:
                    continue
                common_words.add(cols[0].lower())
                if len(common_words) > 500000:
                    break
        vocab = {}
        # +1 for UNK +1 for PAD
        num_words = len(common_words) + 4
        logger.info("#words: %d" % num_words)
        vectors = np.zeros(shape=(num_words, vector_size))
        # Vector for UNK, 0 is reserved for PAD
        UNK_ID = num_words - 1
        vocab[DeepCNNAqquRelScorer.UNK] = UNK_ID
        vectors[UNK_ID, :] = np.random.uniform(-0.05, 0.05,
                                               vector_size)
        vocab["ENTITY"] = num_words - 2
        vectors[num_words - 2] = np.random.uniform(-0.05, 0.05,
                                               vector_size)
        vocab["STRTS"] = num_words - 3
        vectors[num_words - 3] = np.random.uniform(-0.05, 0.05,
                                               vector_size)
        vector_index = 1
        for w in gensim_model.vocab:
            if w not in common_words:
                continue
            vocab[w] = vector_index
            vectors[vector_index, :] = gensim_model[w]
            vector_index += 1
        logger.info("Done")
        return vector_size, vocab, vectors

    def learn_model(self, train_queries, num_epochs=2):
        default_sess = tf.get_default_session()
        if default_sess is not None:
            logger.info("Closing previous default session.")
            default_sess.close()
        train_batches = self.create_train_batches(train_queries)
        self.g = tf.Graph()
        with self.g.as_default():
            self.build_deep_model(self.sentence_len, self.embeddings,
                                  self.embedding_size, self.rel_width_len)
            session_conf = tf.ConfigProto(
                allow_soft_placement=True)
            sess = tf.Session(config=session_conf)
            self.sess = sess
            with self.g.device("/gpu:0"):
                with sess.as_default():
                    optimizer = tf.train.AdamOptimizer()
                    grads_and_vars = optimizer.compute_gradients(self.loss)
                    global_step = tf.Variable(0, name="global_step", trainable=False)
                    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                    sess.run(tf.initialize_all_variables())
                    self.saver = tf.train.Saver()

                    def train_step(x_batch, x_rel_batch, y_batch):
                        """
                        A single training step
                        """
                        feed_dict = {
                          self.input_s: x_batch,
                          self.input_r: x_rel_batch,
                          self.input_y: y_batch,
                          self.dropout_keep_prob: 0.5
                        }
                        _, step, loss, probs = sess.run(
                            [train_op, global_step, self.loss, self.probs],
                            feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        #print("{}: step {}, loss {}".format(time_str, step, loss))
                    for n in range(num_epochs):
                        logger.info("Starting epoch %d" % (n + 1))
                        for y_batch, x_batch, x_rel_batch in train_batches:
                            train_step(x_batch, x_rel_batch, y_batch)

    def create_train_batches(self, train_queries, correct_threshold=.5):
        logger.info("Creating train batches.")
        query_batches = []
        for query in train_queries:
            batch = []
            oracle_position = query.oracle_position
            candidates = [x.query_candidate for x in query.eval_candidates]
            has_correct_candidate = False
            for i, candidate in enumerate(candidates):
                if query.eval_candidates[i].evaluation_result.f1 >= correct_threshold \
                        or i + 1 == oracle_position:
                    batch.append((1, candidate))
                    has_correct_candidate = True
                else:
                    batch.append((0, candidate))
            if has_correct_candidate and len(batch) > 1:
                batch_features = self.create_batch_features(batch)
                query_batches.append(batch_features)
        logger.info("Done. %d batches." % len(query_batches))
        return query_batches

    def create_batch_features(self, batch, max_len=250):
        # Sort so that first candidate is correct - needed for training.
        # Ignores that there may be several correct candidates!
        batch = sorted(batch, key=lambda x: x[0], reverse=True)
        batch = batch[:max_len]
        num_candidates = len(batch)
        # How much to add left and right.
        pad = max(self.filter_sizes) - 1
        words = np.zeros(shape=(num_candidates, self.max_query_len + 2 * pad),
                         dtype=int)
        rel_features = np.zeros(shape=(num_candidates, 6 * self.embedding_size),
                                dtype=float)
        labels = np.zeros(shape=(num_candidates, 1))
        for i, (label, candidate) in enumerate(batch):
            text_tokens = feature_extraction.get_query_text_tokens(candidate)
            text_sequence = []
            # Transform to IDs.
            for t in text_tokens:
                if t in self.vocab:
                    text_sequence.append(self.vocab[t])
                else:
                    text_sequence.append(self.UNK_ID)
            if len(text_sequence) > self.max_query_len:
                logger.warn("Max length exceeded: %s. Truncating" % text_sequence)
                text_sequence = text_sequence[:self.max_query_len]
            for j, t in enumerate(text_sequence):
                words[i, pad + j] = t
            relations = candidate.get_relation_names()
            for j, r in enumerate(relations):
                parts = r.strip().split('.')
                for k, p in enumerate(parts[-3:]):
                    rel_vectors = []
                    for w in p.split('_'):
                        if w in self.vocab:
                            rel_vectors.append(self.embeddings[self.vocab[w]])
                        else:
                            rel_vectors.append(self.embeddings[self.UNK_ID])
                    start = (j * 3 + k) * self.embedding_size
                    end = (j * 3 + k + 1) * self.embedding_size
                    rel_features[i, start:end] = np.average(rel_vectors)
            labels[i] = label
        return labels, words, rel_features

    def store_model(self):
        # Store UNK, as well as name of embeddings_source?
        filename = "data/model-dir/tf/" + self.name + "/"
        logger.info("Writing model to %s." % filename)
        #try:
        #    os.remove(self.name)
        #except OSError:
        #    pass
        if not os.path.exists(filename):
           os.makedirs(filename)
        filename += self.name
        logger.info("Writing model to %s." % filename)
        self.saver.save(self.sess, filename, global_step=100)
        logger.info("Done.")

    def load_model(self):
        filename = "data/model-dir/tf/" + self.name + "/"
        logger.info("Loading model from %s." % filename)
        ckpt = tf.train.get_checkpoint_state(filename)
        if ckpt and ckpt.model_checkpoint_path:
            logger.info("Loading model from %s." % filename)
            self.g = tf.Graph()
            with self.g.as_default():
                self.build_deep_model(self.sentence_len, self.embeddings,
                                      self.embedding_size, self.rel_width_len)
                saver = tf.train.Saver()
                session_conf = tf.ConfigProto(
                    allow_soft_placement=True)
                sess = tf.Session(config=session_conf)
                self.sess = sess
                saver.restore(sess, ckpt.model_checkpoint_path)


    def score(self, candidate):
        from ranker import RankScore
        _, words, rel_features = self.create_batch_features([(0, candidate)])
        feed_dict = {
          self.input_s: words,
          self.input_r: rel_features,
          self.dropout_keep_prob: 1.0
        }
        with self.g.as_default():
            with self.g.device("/gpu:0"):
                with self.sess.as_default():
                    result = self.sess.run(
                        [self.probs],
                        feed_dict)
                    probs = result[0]
                    # probs is a matrix: n x c (for n examples and c
                    # classes)
                    return RankScore(probs[0][0])

    def score_multiple(self, score_candidates, batch_size=100):
        """
        Return a lis tof scores
        :param candidates:
        :return:
        """
        from ranker import RankScore
        result = []
        batch = 0
        while True:
            candidates = score_candidates[batch * batch_size:(batch + 1) * batch_size]
            if not candidates:
                break
            _, words, rel_features = self.create_batch_features([(0, c) for c in candidates])
            feed_dict = {
              self.input_s: words,
              self.input_r: rel_features,
              self.dropout_keep_prob: 1.0
            }
            with self.g.as_default():
                with self.g.device("/gpu:0"):
                    with self.sess.as_default():
                        res = self.sess.run(
                            [self.probs],
                            feed_dict)
                        probs = res[0]
                        # probs is a matrix: n x c (for n examples and c
                        # classes)
                        for i in range(probs.shape[0]):
                            result.append(RankScore(probs[i][0]))
            batch += 1
        return result

    def build_deep_model(self, sentence_len, embeddings, embedding_size,
                         rel_width, filter_sizes=[2, 3, 4], num_filters=200,
                         n_hidden_nodes_1=100, num_classes=1):
        logger.info("sentence_len: %s"% sentence_len)
        logger.info("embedding_size: %s"% embedding_size)
        logger.info("rel_width: %s"% rel_width)

        self.input_s = tf.placeholder(tf.int32, [None, sentence_len], name="input_s")
        self.input_r = tf.placeholder(tf.float32, [None, rel_width], name="input_r")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.margin = tf.constant(1.0)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                embeddings.astype(np.float32),
                name="W",
                trainable=False)
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_s)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sentence_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        self.rh_pool = tf.concat(1, [self.h_drop, self.input_r])

        pooled_width = num_filters_total + rel_width

        with tf.name_scope("dense"):
            W = tf.Variable(tf.truncated_normal([pooled_width, n_hidden_nodes_1],
                                                stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[n_hidden_nodes_1]), name="b")
            self.h_1 = tf.nn.xw_plus_b(self.rh_pool, W, b, name="h_1")
            self.a_1 = tf.nn.relu(self.h_1)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([n_hidden_nodes_1, num_classes],
                                                stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            W = tf.clip_by_norm(W, 3)
            self.scores = tf.nn.xw_plus_b(self.a_1, W, b, name="scores")
            self.probs = tf.nn.sigmoid(self.scores)

        correct_score = self.scores[0, :]
        wrong_scores = self.scores[1:, :]

        def get_rank(correct_score, wrong_scores):
            rank = tf.reduce_sum(tf.cast(tf.greater(tf.add(wrong_scores, self.margin),
                                                    correct_score),
                                         tf.float32))
            # Above behaves weird (negative rank) if wrong_scores is empty
            rank = tf.maximum(0.0, rank)
            return rank

        def rank_loss(rank):
            #return tf.cast(rank, tf.float32)
            rank_int = tf.cast(rank, tf.int32)
            a = tf.cast(tf.range(1, limit=rank_int + 1), tf.float32)
            ones = tf.ones_like(a, dtype=tf.float32)
            l = tf.reduce_sum(tf.div(ones, a), name="rank_loss")
            return l
            #loss = 0
            #for j in range(rank[0]):
            #    loss += 1/j
            #return loss

        rank = get_rank(correct_score, wrong_scores)
        r_loss = rank_loss(rank)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.maximum(0.0, self.margin + wrong_scores - correct_score)
            self.loss = tf.reduce_sum(r_loss * losses / rank)
            #self.loss = tf.reduce_mean(losses)
