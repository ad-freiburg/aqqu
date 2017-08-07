import logging
import math
import random
import numpy as np
import joblib
from sklearn import utils
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

class DeepCNNAqquDSScorer():

    UNK = '---UNK---'
    PAD = '---PAD---'


    def __init__(self, embedding_file):
        self.name = "DSDeepRelScorer"
        self.n_rels = 1
        self.n_parts_per_rel = 3
        self.n_rel_parts = self.n_rels  * self.n_parts_per_rel
        if embedding_file is not None:
            [self.embedding_size, self.vocab,
             self.embeddings] = self.extract_vectors(embedding_file)
            self.UNK_ID = self.vocab[DeepCNNAqquDSScorer.UNK]
            self.PAD_ID = self.vocab[DeepCNNAqquDSScorer.PAD]
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
        num_words = len(gensim_model.vocab) + 4
        logger.info("#words: %d" % num_words)
        vectors = np.zeros(shape=(num_words, vector_size), dtype=np.float32)
        # Vector for UNK, 0 is reserved for PAD
        PAD_ID = 0
        vocab[DeepCNNAqquDSScorer.PAD] = PAD_ID

        # Vector for UNK
        UNK_ID = 1
        vocab[DeepCNNAqquDSScorer.UNK] = UNK_ID
        vectors[UNK_ID] = np.random.uniform(-0.05, 0.05,
                                            vector_size)
        NUM_ID = 2
        vocab["<num>"] = NUM_ID
        vectors[NUM_ID] = np.random.uniform(-0.05, 0.05,
                                            vector_size)
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

    def extend_vocab_for_relwords(self, relations):
        logger.info("Extending vocabulary with words of relations.")
        relation_set = set(relations)
        #additional_rels_file = "rels"
        #logger.info("Reading additional relations from %s" % additional_rels_file)
        #additional_rels = open(additional_rels_file, "r").readlines()
        #relation_set.update([r.strip() for r in additional_rels])
        new_vectors = []
        new_words = []
        for relation in relation_set:
            rel_words = self.split_relation(relation)
            for rel_w_part in rel_words:
                if rel_w_part not in self.vocab:
                    new_vector = self.generate_relation_word(rel_w_part)
                    new_vectors.append(new_vector)
                    new_words.append(rel_w_part)
        for w, v in zip(new_words, new_vectors):
            next_id = len(self.vocab)
            self.vocab[w] = next_id
        self.embeddings = np.vstack((self.embeddings, np.array(new_vectors)))
        self.embeddings = self.embeddings.astype(np.float32)
        #self.embeddings = normalize(self.embeddings, norm='l2', axis=1)
        logger.debug("Added the following words: %s" % str(new_words))
        logger.info("Final final vocabulary size: %d." % len(self.vocab))

    def generate_relation_word(self, rel_w_part):
        parts = rel_w_part.split('_')
        vectors = []
        for w in parts:
            if w not in self.vocab:
                vectors.append(np.random.uniform(-0.05, 0.05,
                                                 self.embedding_size))
            else:
                vectors.append(self.embeddings[self.vocab[w]])
        new_vector = np.average(np.array(vectors), axis=0)
        return new_vector

    def learn_model(self, examples_file, num_epochs=1,
                    dev_ratio=0.1,
                    batch_size=100):
        examples, relations = self.read_training_data(examples_file)
        self.extend_vocab_for_relwords(relations)
        labels, examples, relations = self.add_negative_samples(examples,
                                                                relations)
        word_features, rel_features = self.extract_features(examples, relations)
        labels, examples, relations, word_features, rel_features = utils.shuffle(labels, examples,
                                                                                         relations, word_features,
                                                                                         rel_features)
        labels = np.array(labels).reshape((len(labels), 1))
        num_dev = int(len(labels) * dev_ratio)
        train_labels = labels[:-num_dev]
        dev_labels = labels[-num_dev:]
        train_word_features = word_features[:-num_dev]
        dev_word_features = word_features[-num_dev:]
        train_rel_features = rel_features[:-num_dev]
        dev_rel_features = rel_features[-num_dev:]
        self.g = tf.Graph()
        with self.g.as_default():
            self.build_deep_model(self.sentence_len, self.embeddings,
                                  self.embedding_size, self.n_rel_parts,
                                  filter_sizes=self.filter_sizes)
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.g.device("/gpu:0"):
                with self.sess.as_default():
                    optimizer = tf.train.AdamOptimizer()
                    global_step = tf.Variable(0, name="global_step", trainable=False)
                    train_op = optimizer.minimize(self.loss)
                    self.sess.run(tf.initialize_all_variables())
                    self.saver = tf.train.Saver()

                    def run_dev_batches(batch_size=200):

                        d_probs = []
                        d_labels = []
                        for batch_num, batch in self.batch_iter(batch_size,
                                                                True,
                                                                dev_labels,
                                                                dev_word_features,
                                                                dev_rel_features):
                            b_labels, b_word_features, b_rel_features = batch
                            feed_dict = {
                                self.input_s: b_word_features,
                                self.input_r: b_rel_features,
                                self.input_y: b_labels,
                                self.dropout_keep_prob: 1.0
                            }
                            loss, p = self.sess.run(
                                [self.loss, self.probs],
                                feed_dict)
                            d_probs += [p[i, 0] for i in range(p.shape[0])]
                            d_labels += [l for l in b_labels]

                        tp = 0.01
                        fp = 0.01
                        tn = 0.01
                        fn = 0.01
                        for p, y in zip(d_probs, d_labels):
                            if p > 0.5:
                                if y[0] == 1:
                                    tp += 1
                                elif y[0] == 0:
                                    fp += 1
                            elif p < 0.5:
                                if y[0] == 1:
                                    fn += 1
                                elif y[0] == 0:
                                    tn += 1
                        n_correct = tp + tn
                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                        f1 = 2 * precision * recall / (precision + recall)
                        logger.info("Dev accuracy: %.4f" % (n_correct / len(dev_labels)))
                        logger.info("Dev precision: %.4f recall: %.4f f1: %.4f" % (precision, recall, f1))

                    def train_step(batch,
                                   n_batch):
                        """
                        A single training step
                        """
                        b_labels, b_word_features, b_rel_features = batch
                        feed_dict = {
                          self.input_s: b_word_features,
                          self.input_r: b_rel_features,
                          self.input_y: b_labels,
                          self.dropout_keep_prob: 0.5
                        }
                        _, step, loss, probs = self.sess.run(
                            [train_op, global_step, self.loss, self.probs],
                            feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        if n_batch % 200 == 0:
                            print("{}: batch {}, loss {}".format(time_str, n_batch, loss))
                        return loss
                    logger.info("Training on %d examples with batchsize %d" % (len(train_labels),
                                                                               batch_size))
                    for epoch in range(num_epochs):
                        for batch_num, batch in self.batch_iter(batch_size,
                                                                True,
                                                                train_labels,
                                                                train_word_features,
                                                                train_rel_features):
                            train_step(batch, batch_num)
                            if batch_num % 20000 == 0:
                                run_dev_batches()
                        run_dev_batches()

    def add_negative_samples(self, examples, relations, num_neg=5):
        logger.info("Adding negative examples.")
        all_labels = []
        all_examples = []
        all_relations = []
        relation_set = set(relations)
        # Add an unknown relation so that the network learns to handle those.
        #relation_set.add("UNKNOWN-RELATION+UNKNOWN-RELATION+UNKNOWN-RELATION")
        relation_set.add("UNKNOWN-RELATION")
        for example, relation in zip(examples, relations):
            all_labels.append(1)
            all_examples.append(example)
            all_relations.append(relation)
            neg_rels = random.sample(relation_set, num_neg)
            for neg_rel in neg_rels:
                if neg_rel != relation:
                    all_labels.append(0)
                    all_examples.append(example)
                    all_relations.append(neg_rel)
        logger.info("#examples after adding negative examples: %d" % len(all_labels))
        return all_labels, all_examples, all_relations

    def read_training_data(self, input_file, correct_threshold=.8):
        logger.info("Reading examples from %s" % input_file)
        examples = []
        relations = []
        with open(input_file, "r") as f:
            for line in f:
                cols = line.strip().split('\t')
                example, relation = cols[0], cols[1]
                #if score < correct_threshold:
                #    continue
                examples.append(example)
                relations.append(relation)
        logger.info("Read %d examples." % len(examples))

        return examples, relations

    def extract_features(self, examples, relations):
        num_examples = len(examples)
        logger.info("Extracting features for %d exmaples." % num_examples)
        # How much to add left and right.
        pad = max(self.filter_sizes) - 1
        word_features = np.zeros(
            shape=(num_examples, self.max_query_len + 2 * pad),
            dtype=int)
        rel_features = np.zeros(shape=(num_examples,
                                       self.n_rel_parts), dtype=int)
        oov_words = set()
        n_too_long = 0
        for i, (example, relation) in enumerate(zip(examples, relations)):
            # batch = batch[:max_len]
            text_tokens = example.lower().split(' ')
            text_sequence = []
            # Transform to IDs.
            for t in text_tokens:
                if t in self.vocab:
                    text_sequence.append(self.vocab[t])
                else:
                    #oov_words.add(t)
                    text_sequence.append(self.UNK_ID)
            if len(text_sequence) > self.max_query_len:
                n_too_long += 1
                logger.debug(
                    "Max length exceeded: %s. Truncating" % text_sequence)
                text_sequence = text_sequence[:self.max_query_len]
            for j, t in enumerate(text_sequence):
                word_features[i, pad + j] = t
            rel_words = self.split_relation(relation)
            for j, w in enumerate(rel_words):
                if w in self.vocab:
                    rel_features[i, j] = self.vocab[w]
                else:
                    rel_features[i, j] = self.UNK_ID
                    oov_words.add(w)
        if n_too_long > 0:
            logger.info("%d examples were truncated because they were too long" % n_too_long)
        if oov_words:
            logger.debug("OOV words in batch: %s" % str(oov_words))
        return word_features, rel_features

    def split_relation(self, relation):
        """Split the relation into a list of words.
        domain.sub_domain.rel_name -> [[domain], [sub, domain], [rel, name]]

        :param relation_name:
        :return:
         """
        words = []
        for k, rel in enumerate(relation.split('+')):
            parts = rel.strip().split('.')
            for i, p in enumerate(parts[-self.n_parts_per_rel:]):
                words.append(p)
        n_rels = len(relation.split('+'))
        #if n_rels * self.n_parts_per_rel != len(words):
        #    logger.info("not enough rel parts: %s" % relation)
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
        self.UNK_ID = self.vocab[DeepCNNAqquDSScorer.UNK]
        self.PAD_ID = self.vocab[DeepCNNAqquDSScorer.PAD]
        self.rel_width_len = self.n_rel_parts * self.embedding_size
        logger.info("Loading model from %s." % filename)
        ckpt = tf.train.get_checkpoint_state(filename)
        if ckpt and ckpt.model_checkpoint_path:
            logger.info("Loading model from %s." % filename)
            self.g = tf.Graph()
            with self.g.as_default():
                self.build_deep_model(self.sentence_len, self.embeddings,
                                      self.embedding_size, self.n_rel_parts,
                                      filter_sizes=self.filter_sizes)
                saver = tf.train.Saver()
                session_conf = tf.ConfigProto(
                    allow_soft_placement=True)
                sess = tf.Session(config=session_conf)
                self.sess = sess
                saver.restore(sess, ckpt.model_checkpoint_path)

    def score_multiple(self, score_candidates, batch_size=100):
        """
        Return a lis tof scores
        :param candidates:
        :return:
        """
        from ranker import RankScore
        examples = []
        relations = []
        for i, candidate in enumerate(score_candidates):
            text = " ".join(feature_extraction.get_query_text_tokens(candidate)[1:])
            #if text.endswith(' ?'):
            #    text = text[:-2]
            #text = text.replace("who ", "<entity> ")
            #text = text.replace("where ", "<entity> ")
            #text = text.replace("when ", "<num> ")
            c_relations = candidate.get_unsorted_relation_names()
            if len(c_relations) > 1:
                relation = "UNKNOWN-RELATION"
            else:
                relation = c_relations[0]
                #relation = "+".join(sorted(relations[0]))
            examples.append(text)
            relations.append(relation)
        #open("text", "w").write("\n".join(examples))
        w_features, r_features = self.extract_features(examples, relations)
        result = []
        for batch_num, batch in self.batch_iter(batch_size,
                                                False,
                                                w_features,
                                                r_features):
            b_word_features, b_rel_features = batch
            feed_dict = {
                self.input_s: b_word_features,
                self.input_r: b_rel_features,
                self.dropout_keep_prob: 1.0
            }
            with self.g.as_default():
                with self.g.device("/gpu:0"):
                    with self.sess.as_default():
                        res = self.sess.run(
                            [self.probs],
                            feed_dict)
                        probs = res[0]
                        for i in range(probs.shape[0]):
                            result.append(RankScore(round(probs[i][0], 4)))
        assert(len(result) == len(score_candidates))
        scores = []
        for e, r, s in zip(examples, relations, result):
            scores.append((e, r, s.score))
        scores = sorted(scores, key=lambda x: x[2], reverse=True)
        text = "\n".join(["%s\t%s\t%s" % (e, r, s) for e, r, s in scores])
        open("text", "a").write(text + "\n")
        return result


    def score_custom(self, examples, relations, batch_size=100):
        """
        Return a lis tof scores
        :param candidates:
        :return:
        """
        from ranker import RankScore
        w_features, r_features = self.extract_features(examples, relations)
        result = []
        for batch_num, batch in self.batch_iter(batch_size,
                                                False,
                                                w_features,
                                                r_features):
            b_word_features, b_rel_features = batch
            feed_dict = {
                self.input_s: b_word_features,
                self.input_r: b_rel_features,
                self.dropout_keep_prob: 1.0
            }
            with self.g.as_default():
                with self.g.device("/gpu:0"):
                    with self.sess.as_default():
                        res = self.sess.run(
                            [self.probs],
                            feed_dict)
                        probs = res[0]
                        for i in range(probs.shape[0]):
                            result.append(round(probs[i][0], 4))
        return result

    def build_deep_model(self, sentence_len, embeddings, embedding_size,
                         rel_len, filter_sizes=(2, 3, 4),
                         num_filters=200,
                         n_hidden_nodes_1=400,
                         n_hidden_nodes_2=200,
                         n_hidden_nodes_3=50,
                         num_classes=1):
        logger.info("sentence_len: %s"% sentence_len)
        logger.info("embedding_size: %s"% embedding_size)
        logger.info("rel_len: %s" % rel_len)

        self.input_s = tf.placeholder(tf.int32, [None, sentence_len], name="input_s")
        self.input_r = tf.placeholder(tf.int32, [None, rel_len], name="input_r")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.weight_y = tf.placeholder(tf.float32, [None, num_classes], name="weight_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.margin = tf.constant(1.0)

        with tf.device('/cpu:0'), tf.name_scope("embedding_sentence"):
            W = tf.Variable(
                embeddings,
                name="W",
                trainable=False)
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_s)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)


        with tf.device('/cpu:0'), tf.name_scope("embedding_relation"):
            W = tf.Variable(
                embeddings,
                name="W",
                trainable=False)
            self.embedded_rel = tf.nn.embedding_lookup(W, self.input_r)
            self.embedded_rel_expanded = tf.expand_dims(self.embedded_rel, -1)

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

        rel_flat = tf.reshape(self.embedded_rel_expanded,
                              [-1, self.embedding_size * rel_len])

        self.rh_pool = tf.concat(1, [self.h_drop, rel_flat])

        pooled_width = num_filters_total + rel_len * self.embedding_size

        with tf.name_scope("dense1"):
            W = tf.Variable(tf.truncated_normal([pooled_width, n_hidden_nodes_1],
                                                stddev=0.1, seed=234), name="W")
            self.W_1 = W
            b = tf.Variable(tf.constant(0.1, shape=[n_hidden_nodes_1]), name="b")
            self.h_1 = tf.nn.xw_plus_b(self.rh_pool, W, b, name="h_1")
            self.a_1 = tf.nn.relu(self.h_1)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([n_hidden_nodes_1, num_classes],
                                                stddev=0.1, seed=234), name="W")
            self.W_o = W
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            #W = tf.clip_by_norm(W, 3)
            self.scores = tf.nn.xw_plus_b(self.a_1, W, b, name="scores")
            self.probs = tf.nn.sigmoid(self.scores)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(self.scores,
                                                             self.input_y)
            self.loss = tf.reduce_mean(losses)

def learn():
    #cool = DeepCNNAqquDSScorer("/home/haussmae/qa-completion/data/vectors/entity_sentences_medium.txt_model_128_hs1_neg20_win5")
    cool = DeepCNNAqquDSScorer("/home/haussmae/qa-completion/data/vectors/entity_sentences_medium.txt_model_128_hs1_sg1_neg20_win5")
    cool.learn_model("/home/haussmae/aqqu-bitbucket/data/wikipedia_sentences/ds_30m_train")
    cool.store_model()

def test():
    cool = DeepCNNAqquDSScorer(None)
    cool.load_model()
    texts = ["<entity> is <entity> father",
             "<entity> was born in <entity>  , son of Emily and Alexander Mark MacDonald",
             "<entity> was born in <entity>",
             "<entity> was <entity> born",
             "<entity> where <entity> was born",
             "<entity> was buried in <entity>, where he <entity>, <entity>",
             "where was <entity> born <entity>",
             "<entity> was <entity> born",]
    relations = ["people.person.mother",
                 "people.person.place_of_birth",
                 "people.person.place_of_birth",
                 "people.person.place_of_birth",
                 "people.person.place_of_birth",
                 "people.person.buried",
                 "people.person.place_of_birth",
                 "people.person.religion"]
    scores = cool.score_custom(texts,
                               relations)
    for t, r, s in zip(texts, relations, scores):
        print("%s\t%s\t%.4f" % (t, r, s))


if __name__ == '__main__':
    learn()
    #test()
