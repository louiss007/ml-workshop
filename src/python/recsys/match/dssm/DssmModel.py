# -*- coding=utf-8 -*-
import time
import sys
import numpy as np
import tensorflow as tf


class DssmModel(object):

    def __init__(self, input_node, l1_node, l2_node, batch_size, neg, learning_rate):
        self.INPUT_NODE = input_node
        self.LAYER1_NODE = l1_node
        self.LAYER2_NODE = l2_node
        self.BATCH_SIZE = batch_size
        self.NEG = neg
        self.learning_rate = learning_rate
        self.tag_shape = np.array([self.BATCH_SIZE, self.INPUT_NODE])
        self.doc_shape = np.array([self.BATCH_SIZE, self.INPUT_NODE])

    def get_weight_variable(self, shape, stddev, regularizer):
        weights = tf.get_variable(
            "weights", shape,
            initializer=tf.truncated_normal_initializer(stddev=stddev))

        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(weights))
        return weights

    def inference(self, input_tag_tensor, input_doc_tensor, regularizer):
        with tf.name_scope('layer1'):
            layer1_init_range = np.sqrt(6.0 / (self.INPUT_NODE + self.LAYER1_NODE))
            weights = self.get_weight_variable(
                [self.INPUT_NODE, self.LAYER1_NODE], layer1_init_range, regularizer)
            biases = tf.get_variable(
                [self.LAYER1_NODE],
                initializer=tf.constant_initializer(0.0))

            # tag_layer1 = tf.nn.relu(tf.matmul(input_tag_tensor, weights) + biases)
            # doc_layer1 = tf.nn.relu(tf.matmul(input_doc_tensor, weights) + biases)
            tag_layer1 = tf.nn.relu(tf.sparse_tensor_dense_matmul(input_tag_tensor, weights) + biases)
            doc_layer1 = tf.nn.relu(tf.sparse_tensor_dense_matmul(input_doc_tensor, weights) + biases)

        with tf.name_scope('layer2'):
            layer2_init_range = np.sqrt(6.0 / (self.LAYER1_NODE + self.LAYER2_NODE))

            weights = self.get_weight_variable(
                [self.LAYER1_NODE, self.LAYER2_NODE], layer2_init_range, regularizer)
            biases = tf.get_variable(
                [self.LAYER2_NODE],
                initializer=tf.constant_initializer(0.0)
            )

            tag_layer2 = tf.nn.relu(tf.matmul(tag_layer1, weights) + biases)
            doc_layer2 = tf.nn.relu(tf.matmul(doc_layer1, weights) + biases)

        with tf.name_scope('similarity'):
            # cosine similarity
            query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(tag_layer2), 1, True)), [self.NEG + 1, 1])
            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_layer2), 1, True))

            prod = tf.reduce_sum(tf.mul(tag_layer2, doc_layer2), 1, True)
            norm_prod = tf.mul(query_norm, doc_norm)

            cos_sim = tf.truediv(prod, norm_prod)
            sim = tf.transpose(tf.reshape(tf.transpose(cos_sim), [self.NEG + 1, self.BATCH_SIZE])) * 20
            y = tf.nn.softmax(sim)
            return y

    def loss(self, y, yp):
        with tf.name_scope('loss'):
            hit_prob = tf.slice(prob, [0, 0], [-1, 1])
            loss = -tf.reduce_sum(tf.log(hit_prob)) / BS
            tf.scalar_summary('loss', loss)
            return loss

    def train(self, loss):
        with tf.name_scope('train'):
            # Optimizer
            train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

    def evaluate(self, yp):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(yp, 1), 0)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy', accuracy)

    def save_model(self, sess, model_name, model_path):
        with tf.name_scope('save'):
            saver = tf.train.Saver
            saver.save(sess, model_path+"/{mn}.ckpt".format(mn = model_name))

    def load_model(self, model_path):
        with tf.name_scope('load_model'):
            saver = tf.train.Saver
            model = saver.restore(model_path)
            return model

    def feed_dict(self, is_train, batch_idx):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if is_train:
            query_in, doc_in = pull_batch(query_train_data, doc_train_data, batch_idx)
        else:
            query_in, doc_in = pull_batch(query_test_data, doc_test_data, batch_idx)
        return {query_batch: query_in, doc_batch: doc_in}


    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test', sess.graph)

        # Actual execution
        start = time.time()
        # fp_time = 0
        # fbp_time = 0
        for step in range(FLAGS.max_steps):
            batch_idx = step % FLAGS.epoch_steps
            if batch_idx % FLAGS.pack_size == 0:
                load_train_data(batch_idx / FLAGS.pack_size + 1)

            if batch_idx % (FLAGS.pack_size / 64) == 0:
                progress = 100.0 * batch_idx / FLAGS.epoch_steps
                sys.stdout.write("\r%.2f%% Epoch" % progress)
                sys.stdout.flush()

            sess.run(train_step, feed_dict=feed_dict(True, batch_idx % FLAGS.pack_size))

            if batch_idx == FLAGS.epoch_steps - 1:
                end = time.time()
                epoch_loss = 0
                for i in range(FLAGS.pack_size):
                    loss_v = sess.run(loss, feed_dict=feed_dict(True, i))
                    epoch_loss += loss_v

                epoch_loss /= FLAGS.pack_size
                train_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
                train_writer.add_summary(train_loss, step + 1)

                # print ("MiniBatch: Average FP Time %f, Average FP+BP Time %f" %
                #        (fp_time / step, fbp_time / step))
                #
                print ("\nEpoch #%-5d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" %
                        (step / FLAGS.epoch_steps, epoch_loss, end - start))

                epoch_loss = 0
                for i in range(FLAGS.pack_size):
                    loss_v = sess.run(loss, feed_dict=feed_dict(False, i))
                    epoch_loss += loss_v

                epoch_loss /= FLAGS.pack_size

                test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
                test_writer.add_summary(test_loss, step + 1)

                start = time.time()
                print ("Epoch #%-5d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs" %
                       (step / FLAGS.epoch_steps, epoch_loss, start - end))



