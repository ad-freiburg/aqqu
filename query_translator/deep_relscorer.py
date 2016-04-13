import logging
import math
import random
import numpy as np
import joblib
import sklearn
import numpy as np
import os
import time
import datetime
import tensorflow as tf
from gensim import models
import feature_extraction
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
            self.rel_width_len = self.n_rel_parts * self.embedding_size
        # This is the maximum number of tokens in a query we consider.
        self.max_query_len = 20
        self.filter_sizes = (2, 3, 4)
        self.sentence_len = self.max_query_len + 2 * (max(self.filter_sizes) - 1)

    def extract_vectors(self, gensim_model_fname):
        """Extract vectors from gensim model and add UNK/PAD vectors.
        """
        logger.info("Preparing embeddings matrix")
        np.random.seed(123)
        global gensim_model
        if gensim_model is None:
            gensim_model = models.Word2Vec.load(gensim_model_fname)
        vector_size = gensim_model.vector_size
        common_words = set()
        # Find the most frequent words to keep the embeddings small.
        # TF doesn't work when they are larger than 2GB !?
        #with open("data/word_frequencies.txt") as f:
        #    for line in f:
        #        cols = line.strip().split()
        #        if len(cols) != 2:
        #            continue
        #        common_words.add(cols[0].lower())
        #        if len(common_words) > 500000:
        #            break
        vocab = {}
        # +1 for UNK +1 for PAD, + 1 for ENTITY, + 1 for STRTS
        num_words = len(gensim_model.vocab) + 3
        logger.info("#words: %d" % num_words)
        vectors = np.zeros(shape=(num_words, vector_size), dtype=np.float32)
        # Vector for UNK, 0 is reserved for PAD
        PAD_ID = 0
        vocab[DeepCNNAqquRelScorer.PAD] = PAD_ID

        # Vector for UNK
        UNK_ID = 1
        vocab[DeepCNNAqquRelScorer.UNK] = UNK_ID
        vectors[UNK_ID] = np.random.uniform(-0.05, 0.05,
                                            vector_size)
        #ENTITY_ID = 2
        #vocab["<entity>"] = ENTITY_ID
        #vectors[ENTITY_ID] = np.random.uniform(-0.05, 0.05,
        #                                       vector_size)
        STRTS_ID = 2
        vocab["<start>"] = STRTS_ID
        vectors[STRTS_ID] = np.random.uniform(-0.05, 0.05,
                                              vector_size)
        for w in gensim_model.vocab:
            #if w not in common_words:
            #    continue
            vector_index = len(vocab)
            vocab[w] = vector_index
            vectors[vector_index, :] = gensim_model[w]
        logger.info("Done. Final vocabulary size: %d" % (len(vocab)))
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
        logger.info("Added the following words: %s" % str(new_words))
        logger.info("Final final vocabulary size: %d." % len(self.vocab))

    def evaluate_dev(self, qids, f1s, probs):
        qids, f1s, probs = sklearn.utils.shuffle(qids, f1s, probs,
                                                 random_state=999)
        assert(len(qids) == len(f1s) == len(probs))
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

    def learn_model(self, train_queries, dev_queries, num_epochs=50):
        random.seed(123)
        self.extend_vocab_for_relwords(train_queries)
        np.random.seed(123)
        default_sess = tf.get_default_session()
        if default_sess is not None:
            logger.info("Closing previous default session.")
            default_sess.close()
        if dev_queries is not None:
            dev_features, dev_qids, dev_f1 = self.create_test_batches(dev_queries)
        train_batches = self.create_train_batches(train_queries)
        labels = [l for y_b, _ in train_batches for l in y_b]
        n = float(len(labels))
        n_pos = float(sum(labels))
        n_neg = n - n_pos
        w_pos = n_pos / n
        w_neg = 1 - w_pos
        class_weights = [n / n_neg, n / n_pos]
        self.g = tf.Graph()
        dev_scores = []
        with self.g.as_default():
            tf.set_random_seed(42)
            self.build_deep_model(self.sentence_len, self.embeddings,
                                  self.embedding_size, self.rel_width_len,
                                  filter_sizes=self.filter_sizes)
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.g.device("/gpu:0"):
                with self.sess.as_default():
                    tf.set_random_seed(42)
                    optimizer = tf.train.AdamOptimizer()
                    #optimizer = tf.train.AdagradOptimizer(0.01)
                    #grads_and_vars = optimizer.compute_gradients(self.loss)
                    global_step = tf.Variable(0, name="global_step", trainable=False)
                    #train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                    train_op = optimizer.minimize(self.loss)
                    self.sess.run(tf.initialize_all_variables())
                    self.saver = tf.train.Saver()
                    tf.set_random_seed(42)

                    def run_dev_batches(batch_size=100):
                        n_batch = 0
                        x, x_rel = dev_features
                        num_rows = x.shape[0]
                        probs = []
                        total_loss = 0.0
                        while n_batch * batch_size < num_rows:
                            x_b = x[n_batch * batch_size:(n_batch + 1) * batch_size, :]
                            x_rel_b = x_rel[n_batch * batch_size:(n_batch + 1) * batch_size, :]
                            labels = [1 for _ in range(x_b.shape[0])]
                            weight_y = [class_weights[l] for l in labels]
                            feed_dict = {
                              self.input_s: x_b,
                              self.input_r: x_rel_b,
                              self.input_y: np.array(labels).reshape((x_b.shape[0], 1)),
                              self.weight_y: np.array(weight_y).reshape((x_b.shape[0], 1)),
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
                        logger.info("Dev avg_f1: %.2f oracle_avg_f1: %.2f" % (
                        100 * avg_f1, 100 * oracle_avg_f1))

                    def train_step(x_batch, x_rel_batch, y_batch, n_epoch,
                                   n_batch):
                        """
                        A single training step
                        """
                        #if x_batch.shape[0] > 400:
                        #    logger.info("Truncating batch.")
                        #   rows = [0] + random.sample(range(1, x_batch.shape[0]), 400)
                        #    x_batch = x_batch[rows, :]
                        #    x_rel_batch = x_rel_batch[rows, :]
                        #    y_batch = y_batch[rows, :]
                        weight_y = [class_weights[l] for l in y_batch]
                        feed_dict = {
                          self.input_s: x_batch,
                          self.input_r: x_rel_batch,
                          self.input_y: y_batch,
                          self.weight_y: np.array(weight_y).reshape(
                                (x_batch.shape[0], 1)),
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
                        total_train_loss = 0.0
                        # Need to shuffle the batches in each epoch.
                        logger.info("Starting epoch %d" % (n + 1))
                        random.shuffle(train_batches)
                        batch = 0
                        for y_batch, (x_batch, x_rel_batch) in train_batches:
                            total_train_loss += train_step(x_batch, x_rel_batch, y_batch, n + 1, batch)
                            batch += 1
                        #logger.info("Total train loss: %.2f" % total_train_loss)
                        if dev_queries is not None:
                            run_dev_batches()
                    if dev_scores:
                        logger.info("Dev avg_f1 history:")
                        logger.info(" ".join(["%d:%f" % (i + 1, f)
                                              for i, f in enumerate(dev_scores)]))

    def create_train_batches(self, train_queries, correct_threshold=.5):
        total_num_candidates = len([x.query_candidate for query in train_queries
                                    for x in query.eval_candidates])
        logger.info("Creating train batches from %d queries and %d candidates."
                    % (len(train_queries), total_num_candidates))
        query_batches = []
        total_num_candidates = 0
        total_num_queries = 0
        for query in train_queries:
            batch = []
            oracle_position = query.oracle_position
            candidates = [x.query_candidate for x in query.eval_candidates]
            correct_candidates = []
            for i, candidate in enumerate(candidates):
                f1 = query.eval_candidates[i].evaluation_result.f1
                if f1 >= correct_threshold:
                    #    or i + 1 == oracle_position:
                    correct_candidates.append((f1, 1, candidate))
                else:
                    batch.append((f1, 0, candidate))
            # Avoid noisy examples.
            correct_candidates = sorted(correct_candidates, reverse=True)
            # If there are more than two correct candidates and the margin
            # between 1st and 2nd in F1 is small ignore this question.
            #if len(correct_candidates) > 4 \
            #        and correct_candidates[0][0] - correct_candidates[1][0] < 0.1:
            #    continue
            # Need at least two examples per batch
            if len(correct_candidates) > 0 and len(batch) > 1:
                # Sort so that first candidate is correct - needed for training.
                # Ignores that there may be several correct candidates!
                #batch = sorted(batch, key=lambda x: x[0], reverse=True)
                batch = batch[:1]
                correct_candidates = correct_candidates[:1]

                batch = correct_candidates + batch
                batch = sorted(batch, reverse=True)
                batch_candidates = [b[2] for b in batch]
                batch_labels = [b[1] for b in batch]
                num_candidates = len(batch)
                total_num_candidates += num_candidates
                total_num_queries += 1
                labels = np.array(batch_labels).reshape((num_candidates, 1))
                batch_features = self.create_batch_features(batch_candidates)
                query_batches.append((labels, batch_features))
        logger.info("Done. %d batches/queries, %d candidates." % (total_num_queries,
                                                                  total_num_candidates))
        return query_batches

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
        words = np.zeros(shape=(num_candidates, self.max_query_len + 2 * pad),
                         dtype=int)
        rel_features = np.zeros(shape=(num_candidates,
                                       self.n_rel_parts * self.embedding_size),
                                dtype=float)
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
                logger.warn("Max length exceeded: %s. Truncating" % text_sequence)
                text_sequence = text_sequence[:self.max_query_len]
            for j, t in enumerate(text_sequence):
                words[i, pad + j] = t
            relations = candidate.get_relation_names()
            rel_words = self.split_relation_into_words(relations)
            parts = []
            for ws in rel_words:
                p = []
                for w in ws:
                    if w in self.vocab:
                        p.append(self.embeddings[self.vocab[w]])
                    else:
                        oov_words.add(w)
                if not p:
                    p.append(np.zeros(shape=(self.embedding_size,)))
                parts.append(np.average(np.array(p), axis=0))
            rel_features[i] = np.hstack(parts)
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
        #try:
        #    os.remove(self.name)
        #except OSError:
        #    pass
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
        self.rel_width_len = self.n_rel_parts * self.embedding_size
        logger.info("Loading model from %s." % filename)
        ckpt = tf.train.get_checkpoint_state(filename)
        if ckpt and ckpt.model_checkpoint_path:
            logger.info("Loading model from %s." % filename)
            self.g = tf.Graph()
            with self.g.as_default():
                self.build_deep_model(self.sentence_len, self.embeddings,
                                      self.embedding_size, self.rel_width_len,
                                      filter_sizes=self.filter_sizes)
                saver = tf.train.Saver()
                session_conf = tf.ConfigProto(
                    allow_soft_placement=True)
                sess = tf.Session(config=session_conf)
                self.sess = sess
                saver.restore(sess, ckpt.model_checkpoint_path)

    def score(self, candidate):
        from ranker import RankScore
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
        from ranker import RankScore
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

    def build_deep_model(self, sentence_len, embeddings, embedding_size,
                         rel_width, filter_sizes=(2, 3, 4), num_filters=200,
                         n_hidden_nodes_1=200,
                         n_hidden_nodes_2=200,
                         n_hidden_nodes_3=50,
                         num_classes=1):
        logger.info("sentence_len: %s"% sentence_len)
        logger.info("embedding_size: %s"% embedding_size)
        logger.info("rel_width: %s"% rel_width)

        self.input_s = tf.placeholder(tf.int32, [None, sentence_len], name="input_s")
        self.input_r = tf.placeholder(tf.float32, [None, rel_width], name="input_r")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.weight_y = tf.placeholder(tf.float32, [None, num_classes], name="weight_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.margin = tf.constant(1.0)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                embeddings,
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
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1,
                                                    seed=123),
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

        #pooled_distances = []
        #with tf.name_scope("cosdistance"):
        #    for i in range(0, rel_width, embedding_size):
        #        rel_part = tf.reshape(self.input_r[:, i:i + embedding_size],
        #                              [-1, embedding_size, 1])
        #        product = tf.batch_matmul(self.embedded_chars, rel_part)
        #        product = tf.reshape(product, [-1, sentence_len, 1, 1])
        #        pooled = tf.nn.max_pool(
        #            product,
        #            ksize=[1, sentence_len, 1, 1],
        #            strides=[1, 1, 1, 1],
        #            padding='VALID',
        #            name="pool")
        #        pooled_distances.append(pooled)

        #num_rel_parts = rel_width / embedding_size
        #num_distances_total = num_rel_parts
        #pool_concat = tf.concat(3, pooled_distances)
        #pooled_distances_flat = tf.reshape(tf.concat(3, pooled_distances),
        #                                   [-1, num_distances_total])

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob,
                                        seed=1332)


        #self.rh_pool = tf.concat(1, [self.h_drop, self.input_r])

        pooled_width = num_filters_total

        with tf.name_scope("dense_q"):
            W = tf.Variable(tf.truncated_normal([pooled_width, n_hidden_nodes_1],
                                                stddev=0.1, seed=234), name="W")
            self.W_1 = W
            b = tf.Variable(tf.constant(0.1, shape=[n_hidden_nodes_1]), name="b")
            self.h_1 = tf.nn.xw_plus_b(self.h_drop, W, b, name="h_1")
            self.a_1 = tf.nn.sigmoid(self.h_1)

        with tf.name_scope("dense_r"):
            W = tf.Variable(
                tf.truncated_normal([rel_width, n_hidden_nodes_1],
                                    stddev=0.1, seed=234), name="W")
            self.W_2 = W
            b = tf.Variable(tf.constant(0.1, shape=[n_hidden_nodes_1]),
                            name="b")
            self.h_2 = tf.nn.xw_plus_b(self.input_r, W, b, name="h_2")
            self.a_2 = tf.nn.sigmoid(self.h_2)

        #with tf.name_scope("dense2"):
        #    W = tf.Variable(tf.truncated_normal([n_hidden_nodes_1, n_hidden_nodes_2],
        #                                        stddev=0.1), name="W")
        #    b = tf.Variable(tf.constant(0.1, shape=[n_hidden_nodes_2]), name="b")
        #    self.h_2 = tf.nn.xw_plus_b(self.a_1, W, b, name="h_2")
        #    self.a_2 = tf.nn.relu(self.h_2)


        #with tf.name_scope("dense3"):
        #    W = tf.Variable(tf.truncated_normal([n_hidden_nodes_2, n_hidden_nodes_3],
        #                                        stddev=0.1), name="W")
        #    b = tf.Variable(tf.constant(0.1, shape=[n_hidden_nodes_3]), name="b")
        #    self.h_3 = tf.nn.xw_plus_b(self.a_2, W, b, name="h_3")
        #    self.a_3 = tf.nn.relu(self.h_3)

        # Final (unnormalized) scores and predictions


        #with tf.name_scope("output"):
        #    W = tf.Variable(tf.truncated_normal([n_hidden_nodes_1, num_classes],
        #                                        stddev=0.1, seed=234), name="W")
        #    self.W_o = W
        #    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        #    #W = tf.clip_by_norm(W, 3)
        #    self.scores = tf.nn.xw_plus_b(self.a_1, W, b, name="scores")
        #    self.probs = tf.nn.sigmoid(self.scores)


        norm_a_1 = tf.sqrt(tf.reduce_sum(tf.square(self.a_1), 1, keep_dims=True))
        norm_a_2 = tf.sqrt(tf.reduce_sum(tf.square(self.a_2), 1, keep_dims=True))
        #a_1 = self.a_1 / n
        #a_2 = self.a_2 /norm_a_1
        a_1 = tf.nn.l2_normalize(self.a_1, 1)
        a_2 = tf.nn.l2_normalize(self.a_2, 1)
        scores = tf.mul(a_1, a_2)
        scores = tf.reduce_sum(scores, 1, keep_dims=True)
        self.scores = scores
        self.probs = self.scores

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

        #rank = get_rank(correct_score, wrong_scores)
        #r_loss = rank_loss(rank)

        #self.weighted_logits = tf.mul(self.scores, self.weight_y)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.reduce_sum(wrong_scores) - tf.reduce_sum(correct_score)
            #losses = tf.maximum(0.0, self.margin + wrong_scores - correct_score)
            #losses = tf.nn.sigmoid_cross_entropy_with_logits(self.scores,
            #                                                 self.input_y)
            #self.loss = tf.reduce_sum(r_loss * losses / rank)
            #self.loss = tf.reduce_mean(losses)
                        #0.1 * tf.nn.l2_loss(self.W_o)
            #self.loss = tf.reduce_mean(tf.mul(losses, self.weight_y))
            self.loss = losses
