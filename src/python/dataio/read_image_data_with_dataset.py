import tensorflow as tf


def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'feat1': tf.FixedLenFeature([], tf.int64),
            'feat2': tf.FixedLenFeature([], tf.int64),
        })
    return features['feat1'], features['feat2']


input_files = ["../data/result/image_data/output.tfrecords", "../data/result/image_data/output.tfrecords"]
dataset = tf.data.TFRecordDataset(input_files)

dataset = dataset.map(parser)
iterator = dataset.make_one_shot_iterator()
feat1, feat2 = iterator.get_next()

with tf.Session() as sess:
    for i in range(10):
        f1, f2 = sess.run([feat1, feat2])
        print f1, f2