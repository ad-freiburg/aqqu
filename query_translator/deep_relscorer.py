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
import features


logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

class DeepAqquRelScorer(MLModel):

    def __init__(self, name, embedding_file):
        name += "_DeepRelScorer"
        self.embedding_size = self.embeddings.vector_size
        # Our vector for unknown words
        self.UNK = np.random.uniform(-0.05, 0.05, self.embedding_size)
        # Id for UNK.
        self.UNK_id = len(self.embeddings.vocab)
        self.embeddings = self.extract_vectors(embedding_file)

    def extract_vectors(self, gensim_model_fname):
        logger.info("Preparing embeddings matrix")
        gensim_model = models.Word2Vec.load(model_fname)
        vocab = {}
        # +1 for UNK +1 for PAD
        num_words = len(gensim_model.vocab) + 2
        embedding_size = gensim_model.vector_size
        vectors = np.zeros(shape=(num_words, self.embedding_size))
        vector_index = 0
        for w in self.embeddings:
            vocab[w] = vocab_index
            vectors[vector_index] = self.embeddings[w]
            vector_index += 1


    def learn_model(self, train_queries, correct_threshold=1.0):
        query_batches = []
        for query in queries:
            batch = []
            oracle_position = query.oracle_position
            candidates = [x.query_candidate for x in query.eval_candidates]
            for i, candidate in enumerate(candidates):
                relation = " ".join(candidate.get_relation_names())
                if query.eval_candidates[i].evaluation_result.f1 >= correct_threshold \
                    or i + 1 == oracle_position:
                    batch.append((1, candidate))
                else:
                    batch.append((0, candidate))
            query_batches.append(batch)

    def extract_features_from_candidate(self, candidate):
        text_tokens = features.get_query_text_tokens(candidate)
        text_sequence = []
        # Transform to IDs.
        for t in text_tokens:
            if t in self.embeddings:
                text_sequence.append(self.embeddings.vocab[t].index)


    def store_model(self):
        # Store UNK, as well as name of embeddings_source?
        logger.info("Writing model to %s." % self.get_model_filename())
        joblib.dump([self.model, self.label_encoder,
                     self.dict_vec, self.scaler], self.get_model_filename())
        logger.info("Done.")

    def load_model(self):
        model_file = self.get_model_filename()
        try:
            [model, label_enc, dict_vec, scaler] \
                = joblib.load(model_file)
            self.model = model
            self.dict_vec = dict_vec
            self.scaler = scaler
            self.label_encoder = label_enc
            self.correct_index = label_enc.transform([1])[0]
            logger.info("Loaded scorer model from %s" % model_file)
        except IOError:
            logger.warn("Model file %s could not be loaded." % model_file)
            raise

    def score(self, candidate):
        score = 0.0
        return RankScore(score)

    def build_deep_model(self, sentence_len, embeddings, embedding_size,
                         rel_width,
                         filter_sizes=[2, 3, 4], num_filters=200,
                         n_hidden_nodes_1=200,
                         num_classes=1):

        self.sentence_len = sentence_len
        self.embedding_size = embedding_size
        self.rel_width = rel_width
        self.embeddings = embeddings.astype('float32')
        self.input_s = tf.placeholder(tf.int32, [None, sentence_len], name="input_s")
        self.input_r = tf.placeholder(tf.float32, [None, rel_width], name="input_r")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.margin = tf.constant(1.0)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                self.embeddings,
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



def batch_iter_qid(num_epochs, qids, data):
    """
    Generates a batch iterator for a dataset.
    """
    # qids are tuples of qid - score
    qids = [q[0] for q in qids]
    qids = np.array(qids, dtype='int32')
    unique_qids = np.unique(qids)
    num_unique_ids = len(unique_qids)
    logger.info("Unique qids: %d" % num_unique_ids)
    # A batch consists of one query
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_qid_indices = np.random.permutation(np.arange(num_unique_ids))
        for batch_num in range(num_unique_ids):
            qid = unique_qids[shuffle_qid_indices[batch_num]]
            indices = np.argwhere(qids == qid)
            indices = indices.T[0]
            result = []
            for d in data:
                x = d[indices, :]
                result.append(x)
            yield epoch, batch_num, result


def batch_iter(batch_size, data):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data[0])
    num_batches = math.ceil(data_size/batch_size)
    # qids are tuples of qid - score
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        result = []
        for d in data:
            x = d[start_index:end_index, :]
            result.append(x)
        yield result


def binary_accuracy(probs, y_true):
    predictions = probs > 0.5
    correct = y_true == predictions
    accuracy = np.mean(correct)
    return accuracy

# for reproducibility
np.random.seed(133)
random.seed(133)
infile = "data/aqqu_fancy_rank.pickle"
logger.info("Loading data from %s" % infile)
[X_train, X_rel_train, y_train,
 X_dev, X_rel_dev, y_dev,
 X_test, X_rel_test, y_test,
 vocab_embeddings, vocab, r_vocab,
 q_ids_train,
 q_ids_dev,
 q_ids_test] = joblib.load(
    infile)
rel_width = X_rel_train.shape[1]
max_len = X_train.shape[1]
n_features = len(vocab) + 2
embedding_size = vocab_embeddings.shape[1]
logger.info("Vector dimension: %d" % embedding_size)
logger.info("Max sequence length: %d" % max_len)
logger.info("#features: %s" % n_features)
logger.info("rel_width: %s" % rel_width)


y_train = np.reshape(y_train, (y_train.shape[0], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))
y_dev = np.reshape(y_dev, (y_dev.shape[0], 1))


print(y_train.shape)
print(X_train.shape)
print(X_rel_train.shape)

print(y_dev.shape)
print(X_dev.shape)
print(X_rel_dev.shape)

print(y_test.shape)
print(X_test.shape)
print(X_rel_test.shape)

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.batch_size
FLAGS.embedding_dim = embedding_size

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")



# Training
# ==================================================

g = tf.Graph()
with g.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)

    sess = tf.Session(config=session_conf)
    with g.device("/gpu:0"):
        with sess.as_default():
            cnn = AqquCNN(max_len, vocab_embeddings, embedding_size, rel_width)
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer()
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            #acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, x_rel_batch, y_batch, output_summary=False):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_s: x_batch,
                  cnn.input_r: x_rel_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, probs = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.probs],
                    feed_dict)
                accuracy = binary_accuracy(probs, y_batch)
                if output_summary:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    #train_summary_writer.add_summary(summaries, step)

            def test_step(x_batch, x_rel_batch, y_batch, qids_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                batches = batch_iter(1000, [x_batch, x_rel_batch, y_batch])
                all_probs = []
                for x_b, x_rel_b, y_b in batches:
                    feed_dict = {
                      cnn.input_s: x_b,
                      cnn.input_r: x_rel_b,
                      cnn.input_y: y_b,
                      cnn.dropout_keep_prob: 1.0
                    }
                    step, summaries, loss, probs = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.probs],
                        feed_dict)
                    all_probs.append(probs)
                all_probs = np.vstack(all_probs)
                accuracy = binary_accuracy(all_probs, y_batch)
                avg_f1 = evaluate_queries(qids_batch, all_probs, y_batch)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}, avg_f1 {:g}".format(time_str, step,
                                                                             loss, accuracy, avg_f1))
                if writer:
                    writer.add_summary(summaries, step)
            last_epoch = -1
            # Generate batches
            n_train_queries = len(np.unique(q_ids_train))
            batches = batch_iter_qid(FLAGS.num_epochs, q_ids_train,
                                     [X_train, X_rel_train, y_train])
            # Training loop. For each batch...
            for n_epoch, n_batch, (x_batch, x_rel_batch, y_batch) in batches:
                output_summary = False
                if n_batch % 100 == 0:
                    output_summary = True
                train_step(x_batch, x_rel_batch, y_batch,
                           output_summary=output_summary)
                current_step = tf.train.global_step(sess, global_step)
                if last_epoch != n_epoch:
                    logger.info("Epoch %d, batch %d" % (n_epoch, n_batch))
                    print("\nEvaluation:")
                    test_step(X_test, X_rel_test, y_test, q_ids_test, writer=dev_summary_writer)
                    print("\nEvaluation on dev:")
                    test_step(X_dev, X_rel_dev, y_dev, q_ids_dev, writer=dev_summary_writer)
                    print("")
                    last_epoch = n_epoch
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
            test_step(X_test, X_rel_test, y_test, writer=dev_summary_writer)
            print("")
            last_epoch = n_epoch
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
