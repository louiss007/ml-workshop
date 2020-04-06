# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference_lenet5
import mnist_train_lenet5
import numpy as np

EVAL_INTERVAL_SECS = 20

''' has running problem'''


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            mnist_train_lenet5.BATCH_SIZE,
            mnist_inference_lenet5.IMAGE_SIZE,
            mnist_inference_lenet5.IMAGE_SIZE,
            mnist_inference_lenet5.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(
            tf.float32, [None, mnist_inference_lenet5.OUTPUT_NODE], name='y-input')

        reshaped_x = np.reshape(mnist.validation.images, (mnist_train_lenet5.BATCH_SIZE,
                                                          mnist_inference_lenet5.IMAGE_SIZE,
                                                          mnist_inference_lenet5.IMAGE_SIZE,
                                                          mnist_inference_lenet5.NUM_CHANNELS))
        validate_feed = {x: reshaped_x,
                         y_: mnist.validation.labels}

        y = mnist_inference_lenet5.inference(x, False, None)

        correction_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(
            mnist_train_lenet5.MOVING_AVERAGE_DECAY)

        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # while True:
        for _ in range(5):
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(
                    mnist_train_lenet5.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path \
                    .split('/')[-1].split('-')[-1]

                    accuracy_score = sess.run(accuracy,
                                              feed_dict=validate_feed)
                    print("After %s training step(s), validation "
                          "accuracy = %g" % (global_step, accuracy_score))

                else:
                    print("No checkpoint file found")
                    return

            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("/home/louiss007/MyWorkShop/model/Data/mnist", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()