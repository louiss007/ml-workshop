# -*- coding=utf-8 -*-
import os
import numpy as np
import tensorflow as tf

'''
num_epoch
batch_size
batch_num = sample_num / batch_size
input format: bag of words vector
'''


class DssmModel(object):

    def __init__(self, args):
        self.INPUT_NODE = args.input_node
        self.LAYER1_NODE = args.layer1_node
        self.LAYER2_NODE = args.layer2_node
        self.OUTPUT_NODE = args.output_node
        self.BATCH_SIZE = args.batch_size
        self.NEG = args.neg
        self.learning_rate = args.learning_rate
        self.is_regularizer = args.is_regularizer
        self.regular_factor = args.regular_factor
        self.tag_shape = np.array([self.BATCH_SIZE, self.INPUT_NODE])
        self.doc_shape = np.array([self.BATCH_SIZE, self.INPUT_NODE])

    def get_weight_variable(self, shape, stddev):
        weights = tf.get_variable(
            "weights", shape,
            initializer=tf.truncated_normal_initializer(stddev=stddev))

        if self.is_regularizer:
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.regular_factor)
            tf.add_to_collection('losses', regularizer(weights))
        return weights

    def inference(self, input_tag_tensor, input_doc_tensor):
        with tf.variable_scope('layer1'):
            layer1_init_range = np.sqrt(6.0 / (self.INPUT_NODE + self.LAYER1_NODE))
            weights = self.get_weight_variable(
                [self.INPUT_NODE, self.LAYER1_NODE], layer1_init_range)
            biases = tf.get_variable(
                "biases",
                [self.LAYER1_NODE],
                initializer=tf.constant_initializer(0.0))

            tag_layer1 = tf.nn.relu(tf.matmul(input_tag_tensor, weights) + biases)
            doc_layer1 = tf.nn.relu(tf.matmul(input_doc_tensor, weights) + biases)
            # tag_layer1 = tf.nn.relu(tf.sparse_tensor_dense_matmul(input_tag_tensor, weights) + biases)
            # doc_layer1 = tf.nn.relu(tf.sparse_tensor_dense_matmul(input_doc_tensor, weights) + biases)

        with tf.variable_scope('layer2'):
            layer2_init_range = np.sqrt(6.0 / (self.LAYER1_NODE + self.LAYER2_NODE))

            weights = self.get_weight_variable(
                [self.LAYER1_NODE, self.LAYER2_NODE], layer2_init_range)
            biases = tf.get_variable(
                "biases",
                [self.LAYER2_NODE],
                initializer=tf.constant_initializer(0.0)
            )

            tag_layer2 = tf.nn.relu(tf.matmul(tag_layer1, weights) + biases)
            doc_layer2 = tf.nn.relu(tf.matmul(doc_layer1, weights) + biases)

        with tf.variable_scope('layer3'):
            layer3_init_range = np.sqrt(6.0 / (self.LAYER2_NODE + self.OUTPUT_NODE))

            weights = self.get_weight_variable(
                [self.LAYER2_NODE, self.OUTPUT_NODE], layer3_init_range)
            biases = tf.get_variable(
                "biases",
                [self.OUTPUT_NODE],
                initializer=tf.constant_initializer(0.0)
            )

            tag_layer3 = tf.nn.relu(tf.matmul(tag_layer2, weights) + biases)
            doc_layer3 = tf.nn.relu(tf.matmul(doc_layer2, weights) + biases)

        with tf.variable_scope('similarity'):
            # cosine similarity
            query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(tag_layer3), 1, True)), [self.NEG + 1, 1])
            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_layer3), 1, True))

            prod = tf.reduce_sum(tf.multiply(tag_layer3, doc_layer3), 1, True)
            norm_prod = tf.multiply(query_norm, doc_norm)

            cos_sim = tf.truediv(prod, norm_prod)
            sim = tf.transpose(tf.reshape(tf.transpose(cos_sim), [self.NEG + 1, self.BATCH_SIZE])) * 20
            y = tf.nn.softmax(sim)
        return y

    def save_model(self, sess, model_path, model_name, global_step):
        saver = tf.train.Saver
        saver.save(sess, os.path.join(model_path, model_name), global_step=global_step)

    def train_batch(self, x_tag_batch, x_doc_batch, train_y_batch, is_train):
        y = self.inference(x_tag_batch, x_doc_batch)
        if not is_train:
            return y, None, None
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(train_y_batch, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        train_op = tf.train.GradientDescentOptimizer(self.learning_rate) \
            .minimize(loss)
        return y, loss, train_op

    def evaluate(self, x_tag, x_doc, y_):
        y = self.inference(x_tag, x_doc)
        correction_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))
        return accuracy

    def load_model(self, model_path):
        saver = tf.train.Saver
        model = saver.restore(model_path)
        return model
