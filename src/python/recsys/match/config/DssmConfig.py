'''
parameter config
'''


class DssmConfig():
    input_node = 20000  # default vocab size
    layer1_node = 256
    layer2_node = 512
    output_node = 128
    batch_size = 1024
    neg = 5
    learning_rate = 0.1
    epoch_num = 20  # default epoch
    is_regularizer = False
    regular_factor = 0.1
    is_gpu = True
    sample_num = 0
    input_path_train = "/home/zhengfu/MyWorkShop/dataset/data_for_dssm/train.dat"
    input_path_validation = "/home/zhengfu/MyWorkShop/dataset/data_for_dssm/validation.dat"
    input_path_test = "/home/zhengfu/MyWorkShop/dataset/data_for_dssm/test.dat"
    vocab_config = "/home/zhengfu/MyWorkShop/dataset/data_for_dssm/vocab.txt"
    model_path = "/home/zhengfu/MyWorkShop/dataset/model/"
    model_name = "dssm"

