import tensorflow as tf


train_files = tf.train.match_filenames_once("")
test_files = tf.train.match_filenames_once("")


def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64),
        })
    decoded_image = tf.decode_raw(features['image'], tf.uint8)
    decoded_image.set_shape([features['height'], features['width'], features['channels']])
    label = features['label']
    return decoded_image, label

image_size = 299
batch_size = 128
shuffle_buffer = 10000


dataset = tf.data.TFRecordDataset(train_files)
dataset = dataset.map(parser)

dataset = dataset.map(
    lambda image, label: (
        preprocess_for_train(image, image_size, image_size, None), label))

dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)


NUM_EPOCHS = 10
dataset = dataset.repeat(NUM_EPOCHS)


iterator = dataset.make_initializable_iterator()
image_batch, label_batch = iterator.get_next()

learning_rate = 0.01
logit = inference(image_batch)
loss = calc_loss(logit, label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


test_dataset = tf.data.TFRecordDataset(test_files)
test_dataset = test_dataset.map(parser).map(
    lambda image, label: (
        tf.image.resize_images(image, [image_size, image_size]), label))
test_dataset = test_dataset.batch(batch_size)

test_iterator = test_dataset.make_initializable_iterator()
test_image_batch, test_label_batch = test_iterator.get_next()
test_logit = inference(test_image_batch)
predictions = tf.argmax(test_logit, axis=-1, output_type=tf.int32)

with tf.Session() as sess:
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

    sess.run(iterator.initializer)

    while True:
        try:
            sess.run(train_step)
        except tf.errors.OutOfRangeError:
            break


    sess.run(test_iterator.initializer)
    test_results = []
    test_labels = []
    while True:
        try:
            pred, label = sess.run([predictions, test_label_batch])
            test_results.extend(pred)
            test_labels.extend(label)
        except tf.errors.OutOfRangeError:
            break

correct = [float(y == y_) for (y, y_) in zip(test_labels, test_labels)]
accuracy = sum(correct) / len(correct)
print("Test accuracy is:", accuracy)