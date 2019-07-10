# -*- coding=utf-8 -*-
import numpy as np
import tensorflow as tf


TRAIN_DATA = "../data/text_data/ptb/ptb.train"
EVAL_DATA = "../data/text_data/ptb/ptb.valid"
TEST_DATA = "../data/text_data/ptb/ptb.test"


INPUT_NODE = 1024
LAYER1_NODE = 512
LAYER2_NODE = 256
OUTPUT_NODE = 128

VOCAB_SIZE = 10000
TRAIN_BATCH_SIZE = 512
TRAIN_NUM_STEP = 35

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 5
KEEP_PROB = 0.9
EMBEDDING_KEEP_PROB = 0.9
MAX_GRAD_NORM = 5
SHARE_EMB_AND_SOFTMAX = True


class DssmModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.dropout_keep_prob = KEEP_PROB if is_training else 1.0

    def get_weight_variable(self, shape, regularizer):
        weights = tf.get_variable(
            "weights", shape,
            initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(weights))
        return weights

    def inference(self, input_tensor, regularizer):
        with tf.variable_scope('layer1'):
            weights = self.get_weight_variable(
                [INPUT_NODE, LAYER1_NODE], regularizer)
            biases = tf.get_variable(
                "biases", [LAYER1_NODE],
                initializer=tf.constant_initializer(0.0))
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

        with tf.variable_scope('layer2'):
            weights = self.get_weight_variable(
                [LAYER1_NODE, LAYER2_NODE], regularizer)
            biases = tf.get_variable(
                "biases", [LAYER2_NODE],
                initializer=tf.constant_initializer(0.0))
            layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

        with tf.variable_scope('layer3'):
            weights = self.get_weight_variable(
                [LAYER2_NODE, OUTPUT_NODE], regularizer)
            biases = tf.get_variable(
                "biases", [OUTPUT_NODE],
                initializer=tf.constant_initializer(0.0))
            layer3 = tf.nn.relu(tf.matmul(layer2, weights) + biases)
        return layer3


