import time
from match.config.DssmConfig import *
from run.match_run import *
import codecs

word2id = dict()
vocab_size = 0


def get_word_map(vocab_file):
    global vocab_size
    with codecs.open(vocab_file, 'r', 'utf-8') as fi:
        for n, line in enumerate(fi.readlines()):
            word2id.setdefault(line.strip().lower(), n+1)
    vocab_size = len(word2id)


def get_bow_vec(tag):
    tag_vec = np.zeros(vocab_size)
    for w in tag:
        if w in word2id:
            tag_vec[word2id[w]] += 1
        else:
            tag_vec[word2id['unk']] += 1
    return tag_vec


def get_input_tensor(xs):
    input_tensor = list()
    tag_vec = get_bow_vec(xs[2])
    title_vec = get_bow_vec(xs[0])
    input_tensor.append([tag_vec, title_vec, 1])
    return input_tensor


# bag of word format for tag and title, note: read data with readline, shoud use while loop, not for loop
def load_data_for_dssm(in_file):
    data_X = list()
    start = time.time()
    with codecs.open(in_file, 'r', 'utf-8') as fi:
        while True:
            line = fi.readline()
            if not line:
                break
            lines = line.strip().split("\t ")
            if len(lines) < 3:
                continue
            tensor = get_input_tensor(lines)
            data_X.append(tensor)
    end = time.time()
    print ("load data cost %.2fs" % (end - start))
    return np.reshape(data_X, [len(data_X), 3])


def main(argv=None):
    dssmConfig = DssmConfig()
    if dssmConfig.is_gpu:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

    get_word_map(dssmConfig.vocab_config)
    dssmConfig.input_node = vocab_size
    dataset_train = load_data_for_dssm(dssmConfig.input_path_train)
    sample_num = len(dataset_train)
    dssmConfig.sample_num = sample_num
    assert dssmConfig.sample_num > 0, "number of sample is 0"
    dataset_validation = load_data_for_dssm(dssmConfig.input_path_validation)
    run_train(dataset_train, dataset_validation, dssmConfig)


if __name__ == '__main__':
    tf.app.run()


