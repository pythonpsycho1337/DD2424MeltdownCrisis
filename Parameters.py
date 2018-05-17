#define parameter values and constants

#training parameters
class Training_Parameters:

    def __init__(self):
        self.TRAIN_PERCENT = 0.9
        self.LEARNING_RATE_INIT = 0.1
        self.LEARNING_DECAY = 0.95
        self.DROPOUT = 0.5
        self.BATCH_SIZE = 50
        self.EPOCHS = 5
        self.STEPS = 1 #2000

training_params = Training_Parameters()



#model parameters
class Model_Parameters:

    def __init__(self):

        self.FILTER_SIZES = [3,4,5]
        self.NUM_FILTERS = 100
        self.L2_REG = 3
        self.DENSE_UNITS = 100
        self.RHO = 0.9
        self.NUM_FILTER_GROUPS = 3

    def _set_filter_sizes(self, values_list):
        self.FILTER_SIZES = values_list

    def _set_num_filters(self, number):
        self.NUM_FILTERS = number

network_params = Model_Parameters()

#data parameters
class Data_Parameters:

    def __init__(self):
        self.MAX_SENTENCE_SIZE = 46  #maximum sentence length
        self.NUM_CLASSES = 2
        self.VOCAB_SIZE = 300

data_params = Data_Parameters()

CKPT_PATH = "ckpt/"
SAVE_PATH = "save/"

