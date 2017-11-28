import logging
import math
from itertools import chain
import os
from sklearn import utils
import time
import datetime
import random
import numpy as np
import joblib
import config_helper
import tensorflow as tf
from gensim import models
from . import feature_extraction


logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
gensim_model = None

class DeepCNNAqquRelScorer():

    UNK = '---UNK---'
    PAD = '---PAD---'

    def __init__(self, embeddings_file=None):
        self.n_rels = 3
        self.n_parts_per_rel = 3
        self.n_rel_parts = self.n_rels * self.n_parts_per_rel
        self.embeddings_file = embeddings_file
        # This is the maximum number of tokens in a query we consider.
        self.max_query_len = 20
        self.filter_sizes = (1, 2, 3, 4)
        #pad = max(self.filter_sizes) - 1
        self.sentence_len = self.max_query_len
        # 3 1-word domains, 3 2-word sub domains
        # and 3 3-word relation names plus 2 paddings
        # between domain, sub-domain, relation 20 which is 
        # as long max_query_len
        self.relation_len = 3 + 6 + 9 + 2
        self.scores = None
        self.probs = None
        self.sess = None

    @staticmethod
    def init_from_config():
        """
        Return an instance with options parsed by a config parser.
        :param config_options:
        :return:
        """
        config_options = config_helper.config
        embeddings_file = config_options.get('DeepRelScorer',
                                             'word-embeddings')
        return DeepCNNAqquRelScorer(embeddings_file)

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
        ENTITY_ID = 2
        vocab["<entity>"] = ENTITY_ID
        vectors[ENTITY_ID] = np.random.uniform(-0.05, 0.05,
                                               vector_size)
        STRTS_ID = 3
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

    def extend_vocab_for_relwords(self, examples):
        np.random.seed(234)
        logger.info("Extending vocabulary with words of relations.")
        new_vectors = []
        new_words = []
        for question_tokens, relations in examples:
            rel_splits = self.split_relations_into_words(relations)
            for rel_words in rel_splits:
                for ws in rel_words:
                    for w in ws:
                        if w not in self.vocab:
                            vector = np.random.uniform(-0.05, 0.05,
                                                       self.embedding_size)
                            new_vectors.append(vector)
                            next_id = len(self.vocab)
                            self.vocab[w] = next_id
                            new_words.append(w)
        if new_vectors:
            self.embeddings = \
                np.vstack((self.embeddings, np.array(new_vectors)))
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

    def create_train_examples(self, train_queries, correct_threshold=.5):
        # TODO turn this into a generator and pull it out of the class
        # that will allow training on very large question, relation corpora
        total_num_candidates = len([x.query_candidate
                                    for query in train_queries
                                    for x in query.eval_candidates])
        logger.info("Creating train batches from %d queries and %d candidates."
                    % (len(train_queries), total_num_candidates))
        positive_examples = []
        negative_examples = []

        for query in train_queries:
            candidates = [x.query_candidate for x in query.eval_candidates]
            neg_examples = []
            has_pos_candidate = False
            for i, candidate in enumerate(candidates):
                f1 = query.eval_candidates[i].evaluation_result.f1
                if f1 >= correct_threshold:
                    positive_examples.append((
                        feature_extraction.get_query_text_tokens(candidate),
                        candidate.get_relation_names()))
                    has_pos_candidate = True
                else:
                    neg_examples.append((
                        feature_extraction.get_query_text_tokens(candidate),
                        candidate.get_relation_names()))
            if has_pos_candidate:
                negative_examples.extend(neg_examples)
        return positive_examples, negative_examples

    def create_test_examples(self, test_queries):
        logger.info("Creating test examples.")
        candidate_qids = []
        candidate_f1 = []
        all_candidates = []
        for query in test_queries:
            candidates = [x.query_candidate for x in query.eval_candidates]
            for i, candidate in enumerate(candidates):
                candidate_f1.append(
                    query.eval_candidates[i].evaluation_result.f1)
                candidate_qids.append(query.id)
                all_candidates.append((
                    feature_extraction.get_query_text_tokens(candidate),
                    candidate.get_relation_names()
                    ))
        logger.info(
            "Done. %d batches/queries, %d candidates." % (len(test_queries),
                                                          len(all_candidates)))
        return all_candidates, candidate_qids, candidate_f1


    def learn_model(self, train_candidates, dev_candidates, extend_model=None):
        """
        Wrapper aroung learn_relation_model used to directly take query
        candidates instead of lists of (question_tokens, relations) tuples
        """
        train_pos, train_neg = self.create_train_examples(train_candidates)
        if dev_candidates:
            dev_examples, dev_qids, dev_f1s = \
                self.create_test_examples(dev_candidates)
        else:
            dev_examples, dev_qids, dev_f1s = None, None, None
        self.learn_relation_model(train_pos, train_neg, extend_model,
                                  dev_examples, dev_qids, dev_f1s)


    def init_new_model(self, embeddings, embedding_size):
        """
        Initialize an empty model for learning from scratch
        """

        default_sess = tf.get_default_session()
        if default_sess is not None:
            logger.info("Closing previous default session.")
            default_sess.close()
        self.g = tf.Graph()
        log_name = \
            os.path.join('data/log/', time.strftime("%Y-%m-%d-%H-%M-%S"))
        if not os.path.exists(log_name):
            os.makedirs(log_name)
        self.writer = tf.summary.FileWriter(log_name)

        with self.g.as_default():
            tf.set_random_seed(42)
            self.build_deep_model(embeddings,
                                  embedding_size,
                                  filter_sizes=self.filter_sizes)
        self.writer.add_graph(self.g)

    def learn_relation_model(self, train_pos, train_neg, extend_model=None,
                             dev_examples=None, dev_qids=None, dev_f1s=None,
                             num_epochs=30):
        """
        Learns the model directly on question-relation examples
        """
        random.seed(123)
        if extend_model:
            self.load_model(extend_model)
        elif self.embeddings_file and not extend_model:
            [self.embedding_size, self.vocab,
             self.embeddings] = self.extract_vectors(self.embeddings_file)
            self.extend_vocab_for_relwords(train_pos)
            self.extend_vocab_for_relwords(train_neg)
            self.UNK_ID = self.vocab[DeepCNNAqquRelScorer.UNK]
            self.init_new_model(self.embeddings, self.embedding_size)

        # because extract_vectors sets the seed and isn't always executed
        np.random.seed(123)

        dev_features = None
        if dev_examples:
            dev_features = self.create_batch_features(dev_examples)

        pos_features = self.create_batch_features(train_pos)
        neg_features = self.create_batch_features(train_neg)
        train_pos_word_features = np.array(pos_features[0])
        train_pos_rel_features = np.array(pos_features[1])
        train_neg_word_features = np.array(neg_features[0])
        train_neg_rel_features = np.array(neg_features[1])
        train_pos_labels = \
            np.ones(shape=(len(train_pos), 1), dtype=float)
        train_neg_labels = \
            np.zeros(shape=(len(train_neg), 1), dtype=float)

        dev_scores = []
        with self.g.as_default():
            # when a model was loaded it already inits a session
            if not self.sess:
                gpu_options = tf.GPUOptions(
                    per_process_gpu_memory_fraction=0.9)
                session_conf = tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False,
                    device_count={'GPU': 1},
                    gpu_options=gpu_options)
                self.sess = tf.Session(config=session_conf)

            with self.g.device("/gpu:0"):
                with self.sess.as_default():
                    tf.set_random_seed(42)
                    optimizer = tf.train.AdamOptimizer()
                    # optimizer = tf.train.AdagradOptimizer(0.1)
                    # optimizer = tf.train.RMSPropOptimizer(0.01)
                    # grads_and_vars = optimizer.compute_gradients(self.loss)
                    global_step = \
                        tf.Variable(0, name="global_step", trainable=False)
                    # train_op = \
                    #   optimizer.apply_gradients(grads_and_vars,
                    #                             global_step=global_step)
                    train_op = optimizer.minimize(self.loss)

                    self.sess.run(tf.global_variables_initializer())
                    self.saver = tf.train.Saver(save_relative_paths=True)
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
                                self.dropout_keep_prob: 0.9
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
                            self.dropout_keep_prob: 0.9
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
                        if (n + 1) % 10 == 0 and dev_examples:
                            run_dev_batches(dev_features, dev_qids, dev_f1s, 
                                            dev_train="Dev")
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

    def create_batch_features(self, batch):
        num_questions = len(batch)
        # How much to add left and right.
        words = np.zeros(shape=(num_questions, self.sentence_len),
                         dtype=int)
        rel_features = np.zeros(shape=(num_questions,
                                       self.relation_len),
                                dtype=int)
        oov_words = set()
        for i, example in enumerate(batch):
            text_tokens, relations = example
            text_sequence = []
            # Transform to IDs.
            for t in text_tokens:
                if t in self.vocab:
                    text_sequence.append(self.vocab[t])
                else:
                    oov_words.add(t)
                    text_sequence.append(self.UNK_ID)

            if len(text_sequence) > self.max_query_len:
                logger.debug("Max length exceeded: %s. Truncating",
                            text_sequence)
                text_sequence = text_sequence[:self.max_query_len]

            for word_num, word_id in enumerate(text_sequence):
                words[i, word_num] = word_id

            rel_splits = self.split_relations_into_words(relations)

            rel_sequences = []
            for rel_parts in rel_splits:
                rel_words = []
                for rel_part in rel_parts:
                    word_sequence = []
                    for word in rel_part:
                        if word in self.vocab:
                            word_sequence.append(self.vocab[word])
                        else:
                            oov_words.add(word)
                            word_sequence.append(self.UNK_ID)
                    # tuple so the word sequence is hashable
                    rel_words.append(tuple(word_sequence))
                rel_sequences.append(rel_words)

            # at most 3 relations, and 3 parts per relation
            # so at most 3 groups of 3 elements, each element
            # being all words of that part. Thus:
            # ['a.r.c_d', 'a.x.y', 'a_b.p.q'] ->
            # [ 'a', 'a', 'a', 'b', '', ''
            #   'r', 'x', 'p', '', '', '',
            #   'c', 'd', 'y', 'q', '', '']
            grouped_rel_parts = list(zip(*rel_sequences))

            # For domains and sub-domains we don't
            # care if the same domain is used multiple times
            grouped_rel_parts[0] = list(set(grouped_rel_parts[0]))
            grouped_rel_parts[1] = list(set(grouped_rel_parts[1]))
            # above becomes
            # ['', '', 'a', 'a', 'b', '',
            #  'x', 'r, 'p', '',
            #  'c', 'd', 'y', 'q', '', ''..]
            group_sizes = [3, 6, 9]
            word_num = 0
            for group_num, group in enumerate(grouped_rel_parts):
                flat_group = list(chain(*group))
                for word_id in flat_group[-group_sizes[group_num]:]:
                    rel_features[i, word_num] = word_id
                    word_num += 1
                # one padding between groups
                word_num += 1

        if oov_words:
            logger.debug("OOV words in batch: %s" % str(oov_words))
        return words, rel_features

    def split_relations_into_words(self, relations):
        """Split each of the supplied relations into a list of words.
        domain.sub_domain.rel_name -> [[domain], [sub, domain], [rel, name]]

        :param relations:
        :return:
         """
        split_rels = [[] for _ in range(len(relations))]
        for k, rel in enumerate(relations):
            words = [[] for _ in range(self.n_parts_per_rel)]
            parts = rel.strip().split('.')
            for i, p in enumerate(parts[-self.n_parts_per_rel:]):
                words[i] += p.split('_')
            split_rels[k] = words
        return split_rels

    def store_model(self, path):
        if path[-1] != '/':
            logger.error("Model path needs to end in '/'")
            return
        # Store UNK, as well as name of embeddings_source?
        logger.info("Writing model to %s." % path)
        if not os.path.exists(path):
            os.makedirs(path)

        logger.info("Writing model to %s." % path)
        self.saver.save(self.sess, path, global_step=100)
        vocab_path = path[:-1]+".vocab"
        joblib.dump([self.embeddings, self.vocab, self.embedding_size], vocab_path)
        logger.info("Done.")

    def load_model(self, path):
        if path[-1] != '/':
            logger.error("Model path needs to end in '/'")
            return
        vocab_path = path[:-1] + ".vocab"
        logger.info("Loading model vocab from %s" % vocab_path)
        [self.embeddings, self.vocab, self.embedding_size] = \
            joblib.load(vocab_path)
        self.UNK_ID = self.vocab[DeepCNNAqquRelScorer.UNK]
        logger.info("Loading model from %s." % path)
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            logger.info("Loading model from %s." % path)
            self.g = tf.Graph()

            log_name = os.path.join('data/log/',
                                    time.strftime("%Y-%m-%d-%H-%M-%S"))
            if not os.path.exists(log_name):
                os.makedirs(log_name)
            self.writer = tf.summary.FileWriter(log_name)
            with self.g.as_default():
                self.build_deep_model(self.embeddings,
                                      self.embedding_size,
                                      filter_sizes=self.filter_sizes)
                saver = tf.train.Saver(save_relative_paths=True)
                session_conf = tf.ConfigProto(
                    allow_soft_placement=True)
                self.sess = tf.Session(config=session_conf)
                saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.writer.add_graph(self.g)


    def score(self, candidate):
        from .ranker import RankScore
        words, rel_features = self.create_batch_features([(
            feature_extraction.get_query_text_tokens(candidate),
            candidate.get_relation_names()
            )])
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
        Return a list of scores
        :param candidates:
        :return:
        """
        from .ranker import RankScore
        result = []
        batch = 0
        while True:
            candidates = \
                score_candidates[batch * batch_size:(batch + 1) * batch_size]
            candidate_relations = [(
                feature_extraction.get_query_text_tokens(candidate),
                candidate.get_relation_names()
                ) for candidate in candidates]
            if not candidates:
                break
            words, rel_features = \
                self.create_batch_features(candidate_relations)
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
                         filter_sizes=(2, 3, 4), num_filters=128,
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
                h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")
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
                h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")
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
        self.h_drop = tf.nn.dropout(self.q_h_pool_flat,
                                    self.dropout_keep_prob,
                                    name='dropout_q',
                                    seed=1332)
        self.r_drop = tf.nn.dropout(self.r_h_pool_flat,
                                    self.dropout_keep_prob,
                                    name='dropout_r',
                                    seed=1332)

        pooled_width = num_filters_total

        with tf.name_scope("shared-weights"):
            self.W_shared = tf.Variable(tf.truncated_normal([pooled_width,
                                                             n_hidden_nodes_1],
                                                            stddev=0.1,
                                                            seed=234),
                                        name="W_shared")

        with tf.name_scope("dense_r"):
            b = tf.Variable(tf.constant(0.1, shape=[n_hidden_nodes_1]),
                            name="b")
            self.h_1 = tf.nn.xw_plus_b(self.h_drop, self.W_shared, b,
                                       name="h_1")
            self.a_1 = tf.nn.elu(self.h_1)


        with tf.name_scope("dense_r"):
            b = tf.Variable(tf.constant(0.1, shape=[n_hidden_nodes_1]),
                            name="b")
            self.h_2 = tf.nn.xw_plus_b(self.r_drop, self.W_shared, b,
                                       name="h_2")
            self.a_2 = tf.nn.elu(self.h_2)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            a_1 = tf.nn.l2_normalize(self.a_1, 1, name='normalize_q')
            a_2 = tf.nn.l2_normalize(self.a_2, 1, name='normalize_r')
            scores = tf.multiply(a_1, a_2)
            self.scores = tf.reduce_sum(scores, 1, keep_dims=True)
            self.probs = self.scores
            self.loss = tf.reduce_mean(tf.square(self.input_y - self.scores))

