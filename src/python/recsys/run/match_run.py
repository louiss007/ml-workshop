# -*- coding=utf-8 -*-
from match.dssm.DssmModel import *


def get_next_batch(dataset, idx, batch_size):
    start = idx * batch_size
    end = (idx + 1) * batch_size
    end = end if end < len(dataset) else len(dataset)
    dataset_batch = dataset[start: end]
    x_tag = [x[0] for x in dataset_batch]
    x_doc = [x[1] for x in dataset_batch]
    y = [x[2] for x in dataset_batch]
    return np.array(x_tag), np.array(x_doc), np.array(y)


def run_train(dataset_train, dataset_validation, config):
    x_tag = tf.placeholder(
        tf.float32, [None, config.input_node], name='x-tag-input')
    x_doc = tf.placeholder(
        tf.float32, [None, config.input_node], name='x-doc-input')
    y_ = tf.placeholder(
        tf.float32, [None], name='y-input')

    x_tag_v = tf.placeholder(
        tf.float32, [None, config.input_node], name='x-tag-input-v')
    x_doc_v = tf.placeholder(
        tf.float32, [None, config.input_node], name='x-doc-input-v')
    y_v = tf.placeholder(
        tf.float32, [None], name='y-input-v')

    x_tag_v_in = dataset_validation[:, 0]
    x_doc_v_in = dataset_validation[:, 1]
    y_v_in = dataset_validation[:, 2]
    dssm = DssmModel(config)
    # accuracy = dssm.evaluate(x_tag_v, x_doc_v, y_v)
    batch_num = int(config.sample_num / config.batch_size - 1)
    global_step = tf.Variable(0, trainable=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(config.epoch_num):
            for idx in range(batch_num):
                x_tag_in, x_doc_in, y_in = get_next_batch(dataset_train, idx, config.batch_size)
                yp, loss, train_op = dssm.train_batch(x_tag, x_doc, y_, True)
                _, loss_value, step = sess.run([yp, loss, global_step],
                                            feed_dict={x_tag: x_tag_in, x_doc: x_doc_in, y_: y_in})

                if idx % 100 == 0:
                    print("train step: %d, train loss: %f"
                          "batch step: %d." % (step, loss_value, idx))
                    dssm.save_model(sess, config.model_path, config.model_name, global_step)
            # acc = sess.run(accuracy, feed_dict={x_tag_v: x_tag_v_in, x_doc_v: x_doc_v_in, y_v: y_v_in})
            # print("%d epoch accuracy: %f", i, acc)


# def run_predict(dataset, config):
#     x = tf.placeholder(
#         tf.float32, [None, config.INPUT_NODE], name='x-input')
#     y_ = tf.placeholder(
#         tf.float32, [None, config.OUTPUT_NODE], name='y-input')
#     dssm = DssmModel()
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         ckpt = tf.train.get_checkpoint_state(config.model_path)
#         if ckpt and ckpt.model_checkpoint_path:
#             saver.restore(sess, ckpt.model_checkpoint_path)
