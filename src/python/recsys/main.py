import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', '/model/match/dssm', 'model directory')
flags.DEFINE_float('learning_rate', 0.1, 'initial learning rate.')
flags.DEFINE_integer('max_steps', 900000, 'number of steps of epoch')
flags.DEFINE_integer('epoch_steps', 1000, "number of epoch.")
flags.DEFINE_integer('pack_size', 2000, "Number of batches in one pickle pack.")
flags.DEFINE_bool('gpu', 1, "enable GPU or not")

TRIGRAM_D = 49284

NEG = 50
BATCH_SIZE = 512

INPUT_NODE = TRIGRAM_D
LAYER1_NODE = 400
LAYER2_NODE = 120

config = tf.ConfigProto()  # log_device_placement=True)
config.gpu_options.allow_growth = True
if not FLAGS.gpu:
    config = tf.ConfigProto(device_count={'GPU': 0})

def main():
    def load_data(pack_idx):
        global doc_train_data, query_train_data
        doc_train_data = None
        query_train_data = None
        start = time.time()
        doc_train_data = pickle.load(open('../data/doc.train.' + str(pack_idx) + '.pickle', 'rb')).tocsr()
        query_train_data = pickle.load(open('../data/query.train.' + str(pack_idx) + '.pickle', 'rb')).tocsr()
        end = time.time()
        print ("\nTrain data %d/9 is loaded in %.2fs" % (pack_idx, end - start))