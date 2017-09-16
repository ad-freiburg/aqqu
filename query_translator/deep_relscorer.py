import logging
import math
import random
import numpy as np
import joblib
from sklearn import utils
import config_helper
import os
import time
import datetime
import tensorflow as tf
from gensim import models
from . import feature_extraction
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
gensim_model = None

class DeepCNNAqquRelScorer():

    UNK = '---UNK---'
    PAD = '---PAD---'

    def __init__(self, name, embedding_file):
        self.name = name + "_DeepRelScorer"
        self.n_rels = 3
        self.n_parts_per_rel = 3
        self.n_rel_parts = self.n_rels * self.n_parts_per_rel
        if embedding_file is not None:
            [self.embedding_size, self.vocab,
             self.embeddings] = self.extract_vectors(embedding_file)
            self.UNK_ID = self.vocab[DeepCNNAqquRelScorer.UNK]
        # This is the maximum number of tokens in a query we consider.
        self.max_query_len = 20
        self.filter_sizes = (1, 2, 3)
        pad = max(self.filter_sizes) - 1
        self.sentence_len = self.max_query_len + 2 * pad
        # + 3 to make relation_len=12 a common multiple of the filter
        self.relation_len = self.n_rel_parts + 3
        self.scores = None
        self.probs = None

    @staticmethod
    def init_from_config(name, load_embeddings=True):
        """
        Return an instance with options parsed by a config parser.
        :param config_options:
        :return:
        """
        config_options = config_helper.config
        embeddings_file = None
        if load_embeddings:
            embeddings_file = config_options.get('DeepRelScorer',
                                                 'word-embeddings')
        return DeepCNNAqquRelScorer(name, embeddings_file)

    def extract_vectors(self, gensim_model_fname):
        """Extract vectors from gensim model and add UNK/PAD vectors.
        """
        logger.info("Preparing embeddings matrix")
        np.random.seed(123)
        global gensim_model
        if gensim_model is None:
            gensim_model = models.Word2Vec.load(gensim_model_fname)
        vector_size = gensim_model.vector_size
        vocab = {}
        # +1 for UNK +1 for PAD, + 1 for ENTITY, + 1 for STRTS
        num_words = len(gensim_model.vocab) + 3
        logger.info("#words: %d", num_words)
        vectors = np.zeros(shape=(num_words, vector_size), dtype=np.float32)
        # Vector for PAD, 0 is reserved for PAD
        PAD_ID = 0
        vocab[DeepCNNAqquRelScorer.PAD] = PAD_ID

        # Vector for UNK
        UNK_ID = 1
        vocab[DeepCNNAqquRelScorer.UNK] = UNK_ID
        vectors[UNK_ID] = np.random.uniform(-0.05, 0.05,
                                            vector_size)
        # ENTITY_ID = 2
        # vocab["<entity>"] = ENTITY_ID
        # vectors[ENTITY_ID] = np.random.uniform(-0.05, 0.05,
        #                                       vector_size)
        STRTS_ID = 2
        vocab["<start>"] = STRTS_ID
        vectors[STRTS_ID] = np.random.uniform(-0.05, 0.05,
                                              vector_size)
        #tmin, tmax, tavg = 0.0, 0.0, 0.0
        for w in gensim_model.vocab:
            vector_index = len(vocab)
            vocab[w] = vector_index
            vectors[vector_index, :] = gensim_model[w]
        logger.info("Done. Final vocabulary size: %d", len(vocab))
        #vectors = normalize(vectors, norm='l2', axis=1)
        return vector_size, vocab, vectors

    def extend_vocab_for_relwords(self, train_queries):
        np.random.seed(234)
        logger.info("Extending vocabulary with words of relations.")
        new_vectors = []
        new_words = []
        for query in train_queries:
            candidates = [x.query_candidate for x in query.eval_candidates]
            for candidate in candidates:
                relations = candidate.get_unsorted_relation_names()
                rel_words = self.split_relation_into_words(relations)
                for ws in rel_words:
                    for w in ws:
                        if w not in self.vocab:
                            vector = np.random.uniform(-0.05, 0.05,
                                                       self.embedding_size)
                            new_vectors.append(vector)
                            next_id = len(self.vocab)
                            self.vocab[w] = next_id
                            new_words.append(w)
        self.embeddings = np.vstack((self.embeddings, np.array(new_vectors)))
        self.embeddings = self.embeddings.astype(np.float32)
        #self.embeddings = normalize(self.embeddings, norm='l2', axis=1)
        logger.info("Added the following words: %s", str(new_words))
        logger.info("Final final vocabulary size: %d.", len(self.vocab))

    def evaluate_dev(self, qids, f1s, probs):
        qids, f1s, probs = utils.shuffle(qids, f1s, probs,
                                         random_state=999)
        assert len(qids) == len(f1s) == len(probs)
        queries = {}
        for q, f, p in zip(qids, f1s, probs):
            if q not in queries:
                queries[q] = []
            queries[q].append((p, f))
        all_f1 = 0.0
        all_oracle_f1 = 0.0
        for q, scores in queries.items():
            scores = sorted(scores, key=lambda x: x[0], reverse=True)
            oracle_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            f1 = scores[0][1]
            oracle_f1 = oracle_scores[0][1]
            all_f1 += f1
            all_oracle_f1 += oracle_f1
        num_queries = len(queries)
        logger.info("Evaluating %d queries." % num_queries)
        return all_f1 / num_queries, all_oracle_f1 / num_queries

    def random_sample(self, n, labels, word_features, rel_features):
        indices = np.random.permutation(len(labels))[:n]
        return labels[indices], word_features[indices], rel_features[indices]


    def learn_model(self, train_queries, dev_queries, num_epochs=30):
        random.seed(123)
        self.extend_vocab_for_relwords(train_queries)
        np.random.seed(123)
        default_sess = tf.get_default_session()
        if default_sess is not None:
            logger.info("Closing previous default session.")
            default_sess.close()
        if dev_queries is not None:
            dev_features, dev_qids, dev_f1 = self.create_test_batches(dev_queries)

        pos_labels, pos_features, neg_labels, neg_features = self.create_train_examples(train_queries)
        train_pos_word_features = np.array(pos_features[0])
        train_pos_rel_features = np.array(pos_features[1])
        train_neg_word_features = np.array(neg_features[0])
        train_neg_rel_features = np.array(neg_features[1])
        train_pos_labels = np.array(pos_labels, dtype=float).reshape((len(pos_labels), 1))
        train_neg_labels = np.array(neg_labels, dtype=float).reshape((len(neg_labels), 1))
        self.g = tf.Graph()
        log_name = os.path.join('data/log/', time.strftime("%Y-%m-%d-%H-%M-%S"))
        if not os.path.exists(log_name):
            os.makedirs(log_name)
        self.writer = tf.summary.FileWriter(log_name)

        dev_scores = []
        with self.g.as_default():
            tf.set_random_seed(42)
            self.build_deep_model(self.embeddings,
                                  self.embedding_size,
                                  filter_sizes=self.filter_sizes)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                device_count={'GPU':1},
            gpu_options=gpu_options)
            self.sess = tf.Session(config=session_conf)
            with self.g.device("/gpu:0"):
                with self.sess.as_default():
                    tf.set_random_seed(42)
                    optimizer = tf.train.AdamOptimizer()
                    #optimizer = tf.train.AdagradOptimizer(0.1)
                    #optimizer = tf.train.RMSPropOptimizer(0.01)
                    #grads_and_vars = optimizer.compute_gradients(self.loss)
                    global_step = tf.Variable(0, name="global_step", trainable=False)
                    #train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                    train_op = optimizer.minimize(self.loss)

                    #self.sess.run(tf.initialize_all_variables())
                    self.sess.run(tf.global_variables_initializer())
                    self.saver = tf.train.Saver()
                    tf.set_random_seed(42)

                    def run_dev_batches(dev_features, dev_qids, dev_f1, dev_train,
                                        batch_size=200):
                        n_batch = 0
                        x, x_rel = dev_features
                        num_rows = x.shape[0]
                        probs = []
                        total_loss = 0.0
                        while n_batch * batch_size < num_rows:
                            x_b = x[n_batch * batch_size:(n_batch + 1) * batch_size, :]
                            x_rel_b = x_rel[n_batch * batch_size:(n_batch + 1) * batch_size, :]
                            labels = [1 for _ in range(x_b.shape[0])]
                            input_y = np.array(labels, dtype=float).reshape((len(labels), 1))
                            feed_dict = {
                                self.input_y: input_y,
                                self.input_s: x_b,
                                self.input_r: x_rel_b,
                                self.dropout_keep_prob: 1.0
                            }
                            loss, p = self.sess.run(
                                [self.loss, self.probs],
                                feed_dict)
                            total_loss += loss
                            n_batch += 1
                            probs += [p[i, 0] for i in range(p.shape[0])]
                        avg_f1, oracle_avg_f1 = self.evaluate_dev(dev_qids, dev_f1, probs)
                        dev_scores.append(avg_f1)
                        #logger.info("Dev loss: %.2f" % total_loss)
                        logger.info("%s avg_f1: %.2f oracle_avg_f1: %.2f" % (dev_train,
                        100 * avg_f1, 100 * oracle_avg_f1))

                    def train_step(batch, n_epoch, n_batch):
                        """
                        A single training step
                        """
                        y_batch, x_batch, x_rel_batch = batch
                        feed_dict = {
                            self.input_y: y_batch,
                            self.input_s: x_batch,
                            self.input_r: x_rel_batch,
                            self.dropout_keep_prob: 1.0
                        }
                        _, step, loss, probs = self.sess.run(
                            [train_op, global_step, self.loss, self.probs],
                            feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        if n_batch % 200 == 0:
                            print("{}: step {}, epoch {}, loss {}".format(time_str, n_batch, n_epoch, loss))
                        return loss

                    for n in range(num_epochs):
                        # Need to shuffle the batches in each epoch.
                        logger.info("Starting epoch %d" % (n + 1))

                        n_labels, n_wf, n_rf = self.random_sample(len(train_pos_labels), train_neg_labels, train_neg_word_features, train_neg_rel_features)
                        train_labels = np.vstack([n_labels, train_pos_labels])
                        train_word_features = np.vstack([n_wf, train_pos_word_features])
                        train_rel_featuers = np.vstack([n_rf, train_pos_rel_features])

                        for batch_num, batch in self.batch_iter(50,
                                                                True,
                                                                train_labels,
                                                                train_word_features,
                                                                train_rel_featuers):
                            train_step(batch, n + 1, batch_num)
                        if (n + 1) % 10 == 0 and dev_queries is not None:
                            run_dev_batches(dev_features, dev_qids, dev_f1, dev_train="Dev")
                    if dev_scores:
                        logger.info("Dev avg_f1 history:")
                        logger.info(" ".join(["%d:%f" % (i + 1, f)
                                              for i, f in enumerate(dev_scores)]))

    def batch_iter(self, batch_size, shuffle, *data):
        """
        Generates a batch iterator for a dataset.
        """
        data_size = len(data[0])
        logger.debug("Total examples: %d" % data_size)
        num_batches_per_epoch = int(math.ceil(data_size / float(batch_size)))
        logger.debug("#Batches: %d" % num_batches_per_epoch)
        # Shuffle the data at each epoch
        indices = np.arange(data_size)
        if shuffle:
            indices = np.random.permutation(indices)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            result = []
            for d in data:
                result.append(d[indices[start_index:end_index]])
            yield batch_num, result

    def create_train_examples(self, train_queries, correct_threshold=.5):
        total_num_candidates = len([x.query_candidate for query in train_queries
                                    for x in query.eval_candidates])
        logger.info("Creating train batches from %d queries and %d candidates."
                    % (len(train_queries), total_num_candidates))
        features = []
        total_num_candidates = 0
        total_num_queries = 0
        example_candidates = []
        negative_candidates =[]
        pos_labels = []
        neg_labels = []

        for query in train_queries:
            batch = []
            oracle_position = query.oracle_position
            candidates = [x.query_candidate for x in query.eval_candidates]
            correct_candidates = []
            num_neg = 0
            neg_candidates = []
            has_pos_candidate = False
            for i, candidate in enumerate(candidates):
                f1 = query.eval_candidates[i].evaluation_result.f1
                if f1 >= correct_threshold:
                    example_candidates.append(candidate)
                    pos_labels.append(1)
                    total_num_candidates += 1
                    has_pos_candidate = True
                    total_num_queries += 1
                else:
                    neg_candidates.append(candidate)
                    num_neg += 1
            if has_pos_candidate:
                neg_labels.extend([0 for _ in neg_candidates])
                negative_candidates.extend(neg_candidates)
                total_num_candidates += len(neg_candidates)
        # Avoid noisy examples.
        features = self.create_batch_features(example_candidates)
        neg_features = self.create_batch_features(negative_candidates)
        logger.info("Done. %d batches/queries, %d candidates.", 
                    total_num_queries, total_num_candidates)
        return pos_labels, features, neg_labels, neg_features

    def create_test_batches(self, test_queries):
        logger.info("Creating test batches.")
        candidate_qids = []
        candidate_f1 = []
        all_candidates = []
        for query in test_queries:
            candidates = [x.query_candidate for x in query.eval_candidates]
            all_candidates += candidates
            for i, candidate in enumerate(candidates):
                candidate_f1.append(query.eval_candidates[i].evaluation_result.f1)
                candidate_qids.append(query.id)
        candidate_features = self.create_batch_features(all_candidates,
                                                        max_len=999999)
        logger.info(
            "Done. %d batches/queries, %d candidates." % (len(test_queries),
                                                          len(all_candidates)))
        return candidate_features, candidate_qids, candidate_f1

    def create_batch_features(self, batch, max_len=3000):
        #batch = batch[:max_len]
        num_candidates = len(batch)
        # How much to add left and right.
        pad = max(self.filter_sizes) - 1
        words = np.zeros(shape=(num_candidates, self.sentence_len),
                         dtype=int)
        rel_features = np.zeros(shape=(num_candidates,
                                       self.relation_len),
                                dtype=int)
        oov_words = set()
        for i, candidate in enumerate(batch):
            text_tokens = feature_extraction.get_query_text_tokens(candidate)
            text_sequence = []
            # Transform to IDs.
            for t in text_tokens:
                if t in self.vocab:
                    text_sequence.append(self.vocab[t])
                else:
                    oov_words.add(t)
                    text_sequence.append(self.UNK_ID)

            if len(text_sequence) > self.max_query_len:
                logger.warn("Max length exceeded: %s. Truncating",
                            text_sequence)
                text_sequence = text_sequence[:self.max_query_len]

            for j, t in enumerate(text_sequence):
                words[i, pad + j] = t

            relations = candidate.get_relation_names()
            rel_words = self.split_relation_into_words(relations)
            rel_sequence = []
            for rel_part in rel_words:
                for word in rel_part:
                    if word in self.vocab:
                        rel_sequence.append(self.vocab[word])
                    else:
                        oov_words.add(word)
                        text_sequence.append(self.UNK_ID)

            if len(rel_sequence) > self.n_rel_parts:
                rel_sequence = rel_sequence[-self.n_rel_parts:]

            for j, rel in enumerate(rel_sequence):
                rel_features[i, j] = rel

        if oov_words:
            logger.debug("OOV words in batch: %s" % str(oov_words))
        return words, rel_features

    def split_relation_into_words(self, relations):
        """Split the relation into a list of words.
        domain.sub_domain.rel_name -> [[domain], [sub, domain], [rel, name]]

        :param relation_name:
        :return:
         """
        words = [[] for _ in range(self.n_rel_parts)]
        #words = []
        #words = [[], [], []]
        for k, rel in enumerate(relations):
            parts = rel.strip().split('.')
            for i, p in enumerate(parts[-self.n_parts_per_rel:]):
                words[k * self.n_parts_per_rel + i] += p.split('_')
        return words

    def store_model(self):
        # Store UNK, as well as name of embeddings_source?
        filename = "data/model-dir/tf/" + self.name + "/"
        logger.info("Writing model to %s." % filename)
        if not os.path.exists(filename):
            os.makedirs(filename)

        logger.info("Writing model to %s." % filename)
        self.saver.save(self.sess, filename, global_step=100)
        vocab_filename = "data/model-dir/tf/" + self.name + ".vocab"
        joblib.dump([self.embeddings, self.vocab, self.embedding_size], vocab_filename)
        logger.info("Done.")

    def load_model(self):
        filename = "data/model-dir/tf/" + self.name + "/"
        vocab_filename = "data/model-dir/tf/" + self.name + ".vocab"
        logger.info("Loading model vocab from %s" % vocab_filename)
        [self.embeddings, self.vocab, self.embedding_size] = joblib.load(vocab_filename)
        self.UNK_ID = self.vocab[DeepCNNAqquRelScorer.UNK]
        logger.info("Loading model from %s." % filename)
        ckpt = tf.train.get_checkpoint_state(filename)
        if ckpt and ckpt.model_checkpoint_path:
            logger.info("Loading model from %s." % filename)
            self.g = tf.Graph()

            log_name = os.path.join('data/log/', time.strftime("%Y-%m-%d-%H-%M-%S"))
            if not os.path.exists(log_name):
                os.makedirs(log_name)
            self.writer = tf.summary.FileWriter(log_name)
            with self.g.as_default():
                self.build_deep_model(self.embeddings,
                                      self.embedding_size,
                                      filter_sizes=self.filter_sizes)
                saver = tf.train.Saver()
                session_conf = tf.ConfigProto(
                    allow_soft_placement=True)
                sess = tf.Session(config=session_conf)
                self.sess = sess
                saver.restore(sess, ckpt.model_checkpoint_path)
            self.writer.add_graph(self.g)


    def score(self, candidate):
        from .ranker import RankScore
        words, rel_features = self.create_batch_features([candidate])
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
                    return RankScore(round(probs[0][0], 4))

    def score_multiple(self, score_candidates, batch_size=100):
        """
        Return a lis tof scores
        :param candidates:
        :return:
        """
        from .ranker import RankScore
        result = []
        batch = 0
        while True:
            candidates = score_candidates[batch * batch_size:(batch + 1) * batch_size]
            if not candidates:
                break
            words, rel_features = self.create_batch_features(candidates)
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
                            result.append(RankScore(round(probs[i][0], 4)))
            batch += 1
        assert(len(result) == len(score_candidates))
        return result

    def build_deep_model(self, embeddings, embedding_size,
                         filter_sizes=(2, 3, 4), num_filters=300,
                         n_hidden_nodes_1=200,
                         n_hidden_nodes_r=200,
                         n_hidden_nodes_3=50,
                         num_classes=1):
        logger.info("sentence_len: %s", self.sentence_len)
        logger.info("embedding_size: %s", embedding_size)
        logger.info("n_rel_parts: %s", self.n_rel_parts)

        self.input_s = tf.placeholder(tf.int32, [None, self.sentence_len],
                                      name="input_s")
        self.input_r = tf.placeholder(tf.int32, 
                                      [None, self.relation_len],
                                      name="input_r")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes],
                                      name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                name="dropout_keep_prob")
        self.margin = tf.constant(1.0)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                embeddings,
                name="W",
                trainable=False)
            self.embedded_input_r = tf.nn.embedding_lookup(W, self.input_r)
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_s)
            self.embedded_input_r_expanded = tf.expand_dims(
                    self.embedded_input_r,
                    -1)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars,
                                                          -1)

        # Convolution layers for query text, one conv-maxpool per filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-q-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1,
                                                    seed=123),
                                name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]),
                                name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv-q")
                # Apply nonlinearity
                h = tf.nn.elu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sentence_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool-q")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.q_h_pool = tf.concat(pooled_outputs, 3)
        self.q_h_pool_flat = tf.reshape(self.q_h_pool, [-1, num_filters_total])

        # Convolution layers for relations, one conv-maxpool per filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-r-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1,
                                                    seed=123),
                                name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]),
                                name="b")
                conv = tf.nn.conv2d(
                    self.embedded_input_r_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv-r")
                # Apply nonlinearity
                h = tf.nn.elu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.relation_len - filter_size + 1,
                           1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool-r")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.r_h_pool = tf.concat(pooled_outputs, 3)
        self.r_h_pool_flat = tf.reshape(self.r_h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.q_h_pool_flat,
                                        self.dropout_keep_prob,
                                        name='dropout_q',
                                        seed=1332)
            self.r_drop = tf.nn.dropout(self.r_h_pool_flat,
                                        self.dropout_keep_prob,
                                        name='dropout_r',
                                        seed=1332)


        pooled_width = num_filters_total

        with tf.name_scope("dense_q"):
            W = tf.Variable(tf.truncated_normal([pooled_width,
                                                 n_hidden_nodes_1],
                                                stddev=0.1, seed=234),
                            name="W")
            self.W_1 = W
            b = tf.Variable(tf.constant(0.1, shape=[n_hidden_nodes_1]),
                            name="b")
            self.h_1 = tf.nn.xw_plus_b(self.h_drop, W, b, name="h_1")
            self.a_1 = tf.nn.elu(self.h_1)


        #with tf.name_scope("dense_r_input"):
        #    W = tf.Variable(
        #        tf.truncated_normal([rel_width, n_hidden_nodes_r],
        #                            stddev=0.1, seed=234), name="W")
        #    self.W_3 = W
        #    b = tf.Variable(tf.constant(0.1, shape=[n_hidden_nodes_r]),
        #                    name="b")
        #    self.h_3 = tf.nn.xw_plus_b(self.r_drop, W, b, name="h_3")
        #    self.a_3 = tf.nn.elu(self.h_3)

        with tf.name_scope("dense_r"):
            W = tf.Variable(
                tf.truncated_normal([pooled_width, n_hidden_nodes_1],
                                    stddev=0.1, seed=234), name="W")
            self.W_2 = W
            b = tf.Variable(tf.constant(0.1, shape=[n_hidden_nodes_1]),
                            name="b")
            self.h_2 = tf.nn.xw_plus_b(self.r_drop, W, b, name="h_2")
            self.a_2 = tf.nn.elu(self.h_2)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            a_1 = tf.nn.l2_normalize(self.a_1, 1, name='normalize_q')
            a_2 = tf.nn.l2_normalize(self.a_2, 1, name='normalize_r')
            scores = tf.multiply(a_1, a_2)
            self.scores = tf.reduce_sum(scores, 1, keep_dims=True)
            self.probs = self.scores
            self.loss = tf.reduce_mean(tf.square(self.input_y - self.scores))

        self.writer.add_graph(self.g)
