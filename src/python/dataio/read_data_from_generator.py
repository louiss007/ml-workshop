import tensorflow as tf
import numpy as np

data_numpy = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
dataset = tf.data.Dataset.from_tensors(data_numpy)
import itertools

sess = tf.Session()


def gen():
    for i in itertools.count(1):
        yield (i, [1] * i)


ds = tf.data.Dataset.from_generator(
    gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))
value = ds.make_one_shot_iterator().get_next()

print(sess.run(value))  # (1, array([1]))
print(sess.run(value))  # (2, array([1, 1]))