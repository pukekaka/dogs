

class init_value:
    def __init__(self):
        self.n_classes = 5 # category size
        self.seq_length = 50  # time
        self.read_head_num = 4
        self.batch_size = 128
        self.insts_size = 400
        self.num_epoches = 100000
        self.learning_rate = 1e-3
        self.rnn_size = 200
        self.rnn_num_layers = 1
        self.memory_size = 128
        self.memory_vector_dim = 40
        self.model_dir = 'model'
        self.tensorboard_dir = 'tensorboard'
        # self.n_train_classes = 1200
        self.n_train_classes = 120
        # self.n_test_classes = 423
        self.n_test_classes = 20