from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from train import *
from Parameters import Model_Parameters

network_params = Model_Parameters()  # global variable with network parameters

#optimize input hyperparameter from input distribution
#perform stochastic grid search
#param: filter_size, filter_num
def hyperparameter_optimization(data, param, param_range, dir):

    training_set = data[0:2]
    validation_set = data[2:4]

    accuracy_list = []
    param_list = []
    for i in range(10): #for each parameter value

        params_sampled = np.random.randint(param_range['start'], param_range['end'], network_params.NUM_FILTER_GROUPS)
        param_list.append(params_sampled)

        #change the corresponding network parameter
        if (param == 'filter_size'):
            network_params._set_filter_sizes(params_sampled)
        elif (param == 'filter_num'):
            network_params._set_filter_num(params_sampled)

        valid_accuracy = train(training_set, validation_set, dir)
        accuracy_list.append(valid_accuracy)

    print("\nValidation accuracy of different models: {0:f}\n".format((param_list, accuracy_list)))


















