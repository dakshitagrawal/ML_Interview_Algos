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

    # DONE
    perm = np.random.permutation(len(X))
    shuffle_x = X[perm]
    shuffle_y = y[perm]
    return shuffle_x, shuffle_y


def random_init(shape):
    """
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # DO NOT CHANGE THIS
    np.random.seed(np.prod(shape))

    # DONE
    rand_array = np.random.rand(shape)
    rand_array[0] = 0
    return rand_array


def zero_init(shape):
    """
    Initialize a numpy array of the specified shape with zero
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # DONE
    mat = np.zeros(shape)
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
        # DONE
        self.w1 = weight_init_fn([self.n_input + 1, self.n_hidden])
        self.w2 = weight_init_fn([self.n_hidden + 1, self.n_output])

        # initialize parameters for adagrad
        self.epsilon = 1e-5
        # DONE
        self.grad_sum_w1 = np.zeros(self.w1.shape)
        self.grad_sum_w2 = np.zeros(self.w2.shape)

        # DONE feel free to add additional attributes
        self.out1 = None
        self.act1 = None
        self.act1_bias = None
        self.out2 = None
        self.act2 = None


def sigmoid(x):
    # DONE
    out = 1 / (1 + np.exp(-x))
    return out


def softmax(x):
    # DONE
    x -= x.max(-1, keepdims=True)
    out = np.exp(x)
    out_sum = out.sum(-1, keepdims=True)
    return out / out_sum


def cross_entropy_loss(pred, target):
    # DONE
    loss = -(target * np.log(pred)).sum(-1).mean()
    return loss


def forward(X, nn):
    """
    Neural network forward computation.
    Follow the pseudocode!
    :param X: input data
    :param nn: neural network class
    :return: output probability
    """
    # DONE
    # X --> (N, D+1)
    # w1 --> (D+1, F1)
    nn.out1 = X @ nn.w1  # (N, F1)
    nn.act1 = sigmoid(nn.out1)  # (N, F1)
    nn.act1_bias = np.concatenate(
        (np.ones((len(nn.act1), 1)), nn.act1), axis=-1
    )  # (N, F1+1)
    # w2 --> (F1+1, F2)
    nn.out2 = nn.act1_bias @ nn.w2  # (N, F2)
    nn.act2 = softmax(nn.out2)  # (N, F2)
    return nn.act2


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
    # DONE
    d_out2 = y_hat - y  # (N, F2)
    d_w2 = nn.act1_bias.T @ d_out2  # (F1+1, F2)
    d_act1_bias = d_out2 @ nn.w2.T  # (N, F1+1)
    d_act1 = d_act1_bias[:, 1:]  # (N, F1)
    d_out1 = d_act1 * (1 - nn.act1) * nn.act1  # (N, F1)
    d_w1 = X.T @ d_out1
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
    # DONE
    labels = forward(X, nn).argmax(-1)
    error_rate = (labels != y).sum() / len(y)
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

            # DONE implement adagrad update
            nn.grad_sum_w1 += g_w1 ** 2
            nn.grad_sum_w2 += g_w2 ** 2

            nn.w1 -= g_w1 * nn.lr / np.sqrt(nn.grad_sum_w1 + nn.epsilon)
            nn.w2 -= g_w2 * nn.lr / np.sqrt(nn.grad_sum_w2 + nn.epsilon)

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
