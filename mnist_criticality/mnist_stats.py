
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


class MnistStats(object):
    """
     This class represents the MNIST hand written digit recognition problem
     for measuring statistics describing criticality

     Convolutional Neural Network for MNIST dataset classification task.
        References:
            Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
            learning applied to document recognition." Proceedings of the IEEE,
            86(11):2278-2324, November 1998.
        Links:
            [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
    """

    def __init__(self, params):
        self.params = params if params is not None else {'tensorboard_verbose': 3,
                                                         'tensorboard_dir': "mnist_criticality/tensorboard_dir"}
        self.network = self.build_network()
        self.model = tflearn.DNN(self.network,
                                 tensorboard_verbose=self.params['tensorboard_verbose'],
                                 tensorboard_dir=self.params['tensorboard_dir'],
                                 )

    def build_network(self):
        # Building convolutional network
        network = input_data(shape=[None, 28, 28, 1], name='input')
        network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)
        network = local_response_normalization(network)
        network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)
        network = local_response_normalization(network)
        network = fully_connected(network, 128, activation='tanh')
        network = dropout(network, 0.8)
        network = fully_connected(network, 256, activation='tanh')
        network = dropout(network, 0.8)
        network = fully_connected(network, 10, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=0.01,
                             loss='categorical_crossentropy', name='target')

        return network


