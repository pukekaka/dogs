import os


class init_value:
    def __init__(self):

        current_directory = os.path.dirname(os.path.abspath(__file__))
        home_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
        model_file_path = os.path.join(home_directory, 'model', 'bin_model', 'basicblock_by_file_bin2vec.model')
        label_file_path = os.path.join(home_directory, 'model', 'label_list.csv')
        data_file_path = os.path.join(home_directory, 'output', 'm_dataset')
        tensorboard_directory = os.path.join(home_directory, 'tensorboard')
        save_dir = os.path.join(home_directory, 'model', 'mann_save')

        self.n_classes = 5 # category size
        self.seq_length = 50  # time
        self.read_head_num = 4
        # self.read_head_num = 8
        self.batch_size = 16
        # self.sample_size = 400
        self.sample_size = 400
        self.num_epoches = 100000
        self.learning_rate = 1e-2
        # self.learning_rate = 1e-4
        self.rnn_size = 200
        self.rnn_num_layers = 1
        # self.rnn_num_layers = 2
        self.memory_size = 128
        # self.memory_size = 256
        self.memory_vector_dim = 40
        # self.memory_vector_dim = 80
        self.label_path = label_file_path
        self.model_path = model_file_path
        self.data_path = data_file_path
        self.save_dir = save_dir
        self.tensorboard_dir = tensorboard_directory
        # self.n_train_classes = 112
        # self.n_test_classes = 48
        self.n_train_classes = 10
        self.n_test_classes = 5
