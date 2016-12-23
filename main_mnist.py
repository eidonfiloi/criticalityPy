import tensorflow as tf
from utils.mnist_utils import *
from mnist_criticality.mnist_stats import *
import numpy as np


X, Y, testX, testY = load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

if __name__ == '__main__':
    mnist_runner = MnistStats(params=None)
    mnist_runner.model.fit(
        {'input': X},
        {'target': Y},
        n_epoch=20,
        validation_set=({'input': testX}, {'target': testY}),
        snapshot_step=100,
        show_metric=True,
        run_id='convnet_mnist'
    )

    w = tf.trainable_variables()

    variables = []

    for var in w:
        if "W" in var.name:
            variables.append(np.ravel(mnist_runner.model.get_weights(var)))

    np.save("mnist_criticality/mnist_weights", np.concatenate(variables).ravel())


