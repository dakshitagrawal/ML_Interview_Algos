import argparse

import numpy as np


def shuffle(X, y, epoch):
    """
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    # DO NOT CHANGE THIS
    np.random.seed(epoch)

    # TODO
    shuffle_x = None
    shuffle_y = None
    return shuffle_x, shuffle_y


def random_init(shape):
    """
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # DO NOT CHANGE THIS
    np.random.seed(np.prod(shape))

    # TODO
    rand_array = None
    return rand_array


def zero_init(shape):
    """
    Initialize a numpy array of the specified shape with zero
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # TODO
    mat = None
    return mat


class NN(object):
    def __init__(
        self, lr, n_epoch, weight_init_fn, input_size, hidden_size, output_size
    ):
        """
        Initialization
        :param lr: learning rate
        :param n_epoch: number of training epochs
        :param weight_init_fn: weight initialization function
        :param input_size: number of units in the input layer
        :param hidden_size: number of units in the hidden layer
        :param output_size: number of units in the output layer
        """
        self.lr = lr
        self.n_epoch = n_epoch
        self.weight_init_fn = weight_init_fn
        self.n_input = input_size
        self.n_hidden = hidden_size
        self.n_output = output_size

        # initialize weights and biases for the models
        # HINT: pay attention to bias here
        # TODO
        self.w1 = None
        self.w2 = None

        # initialize parameters for adagrad
        self.epsilon = 1e-5
        # TODO
        self.grad_sum_w1 = None
        self.grad_sum_w2 = None

        # TODO feel free to add additional attributes


def sigmoid(x):
    # TODO
    out = None
    return out


def softmax(x):
    # TODO
    out = None
    return out


def cross_entropy_loss(pred, target):
    # TODO
    loss = None
    return loss


def forward(X, nn):
    """
    Neural network forward computation.
    Follow the pseudocode!
    :param X: input data
    :param nn: neural network class
    :return: output probability
    """
    # TODO
    out = None
    return out


def backward(X, y, y_hat, nn):
    """
    Neural network backward computation.
    Follow the pseudocode!
    :param X: input data
    :param y: label
    :param y_hat: prediction
    :param nn: neural network class
    :return:
    d_w1: gradients for w1
    d_w2: gradients for w2
    """
    # TODO
    d_w1 = None
    d_w2 = None
    return d_w1, d_w2


def test(X, y, nn):
    """
    Compute the label and error rate.
    :param X: input data
    :param y: label
    :param nn: neural network class
    :return:
    labels: predicted labels
    error_rate: prediction error rate
    """
    # TODO
    labels = None
    error_rate = None
    return labels, error_rate


def train(X_tr, y_tr, nn, X_te, onehot_y_te, out_metrics):
    """
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param nn: neural network class
    """
    for epoch in range(nn.n_epoch):
        X_tr_shuffle, y_tr_shuffle = shuffle(X_tr, y_tr, epoch)
        for sample in zip(X_tr_shuffle, y_tr_shuffle):
            y_hat = forward(sample[0][None], nn)
            g_w1, g_w2 = backward(sample[0][None], sample[1][None], y_hat, nn)

            # TODO implement adagrad update

        loss_train = cross_entropy_loss(forward(X_tr_shuffle, nn), y_tr_shuffle)
        loss_test = cross_entropy_loss(forward(X_te, nn), onehot_y_te)

        with open(out_metrics, "a") as f:
            f.write(f"epoch={epoch} crossentropy(train): {loss_train}\n")
            f.write(f"epoch={epoch} crossentropy(validation): {loss_test}\n")


def args2data(parser):
    """
    Parse argument, create data and label.
    :return:
    X_tr: train data (numpy array)
    y_tr: train label (numpy array)
    X_te: test data (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """

    # # Get data from arguments
    out_tr = parser.train_out
    out_te = parser.validation_out
    out_metrics = parser.metrics_out
    n_epochs = parser.num_epoch
    n_hid = parser.hidden_units
    init_flag = parser.init_flag
    lr = parser.learning_rate

    X_tr = np.loadtxt(parser.train_input, delimiter=",")
    y_tr = X_tr[:, 0].astype(int)
    X_tr[:, 0] = 1.0  # add bias terms

    X_te = np.loadtxt(parser.validation_input, delimiter=",")
    y_te = X_te[:, 0].astype(int)
    X_te[:, 0] = 1.0  # add bias terms

    return (
        X_tr,
        y_tr,
        X_te,
        y_te,
        out_tr,
        out_te,
        out_metrics,
        n_epochs,
        n_hid,
        init_flag,
        lr,
    )


def write_error(output_metric_file, train_error, test_error):
    with open(output_metric_file, "a") as f:
        f.write(f"error(train): {train_error}\n")
        f.write(f"error(test): {test_error}\n")


def write_pred(out_file, preds):
    preds = [f"{pred}\n" for pred in preds]
    with open(out_file, "w") as f:
        f.writelines(preds)


def parse_metrics(file, num_lines):
    losses = []
    with open(file, "r") as f:
        for i in range(2 * num_lines):
            line = f.readline()
            if i % 2 == 1:
                losses.append(float(line.split(" ")[-1][:-1]))
    return losses


def parse_metrics_one(file, num_lines):
    train_losses = []
    val_losses = []
    with open(file, "r") as f:
        for i in range(num_lines):
            line = f.readline()
            train_losses.append(float(line.split(" ")[-1][:-1]))
            line = f.readline()
            val_losses.append(float(line.split(" ")[-1][:-1]))
    return train_losses, val_losses


def main(args):
    (
        X_tr,
        y_tr,
        X_te,
        y_te,
        out_tr,
        out_te,
        out_metrics,
        n_epochs,
        n_hid,
        init_flag,
        lr,
    ) = args2data(args)

    num_labels = 10
    onehot_y_tr = np.zeros((y_tr.shape[0], num_labels))
    onehot_y_tr[np.arange(y_tr.shape[0]), y_tr] = 1

    onehot_y_te = np.zeros((y_te.shape[0], num_labels))
    onehot_y_te[np.arange(y_te.shape[0]), y_te] = 1

    weight_init_fn = None
    if args.init_flag == 1:
        weight_init_fn = random_init
    elif args.init_flag == 2:
        weight_init_fn = zero_init

    # Build model
    num_features = X_tr.shape[1] - 1
    my_nn = NN(lr, n_epochs, weight_init_fn, num_features, n_hid, num_labels)

    # train model
    train(X_tr, onehot_y_tr, my_nn, X_te, onehot_y_te, out_metrics)

    # test model and get predicted labels and errors
    pred_train, error_train = test(X_tr, y_tr, my_nn)
    pred_test, error_test = test(X_te, y_te, my_nn)

    # write predicted label and error into file
    write_error(out_metrics, error_train, error_test)
    write_pred(out_tr, pred_train)
    write_pred(out_te, pred_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_input", type=str, help="path to training input .csv file"
    )
    parser.add_argument(
        "validation_input", type=str, help="path to validation input .csv file"
    )
    parser.add_argument(
        "train_out", type=str, help="path to store prediction on training data"
    )
    parser.add_argument(
        "validation_out", type=str, help="path to store prediction on validation data"
    )
    parser.add_argument(
        "metrics_out", type=str, help="path to store training and testing metrics"
    )
    parser.add_argument("num_epoch", type=int, help="number of training epochs")
    parser.add_argument("hidden_units", type=int, help="number of hidden units")
    parser.add_argument(
        "init_flag",
        type=int,
        choices=[1, 2],
        help="weight initialization functions, 1: random",
    )
    parser.add_argument("learning_rate", type=float, help="learning rate")
    args = parser.parse_args()

    main(args)
