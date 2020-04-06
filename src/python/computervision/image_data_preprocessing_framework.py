import tensorflow as tf

from image_preprocessing import preprocess_for_train

files = tf.train.match_filenames_once("data/image_file")
filename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
    })

image, label = features['image'], features['label']
height, width = features['height'], features['width']
channels = features['channels']

decoded_image = tf.decode_raw(image, tf.uint8)
decoded_image.set_shape([height, width, channels])

image_size = 299

distorted_image = preprocess_for_train(
    decoded_image, image_size, image_size, None)

min_after_dequeue = 10000
batch_size = 128
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [distorted_image, label], batch_size=batch_size,
    capacity=capacity, min_after_dequeue=min_after_dequeue)

leaning_rate = 0.01
logit = inference(image_batch)
loss = calc_loss(logit, label_batch)
train_step = tf.train.GradientDescentOptimizer(leaning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    TRAINING_ROUND = 5000
    for i in range(TRAINING_ROUND):
        sess.run(train_step)
    coord.request_stop()
    coord.join(threads)