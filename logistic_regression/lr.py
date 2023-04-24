import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm


def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    out = 1.0 / (1 + np.exp(-x))
    return out


def dJ(theta, X, y, i):
    out = forward(theta, X[i : i + 1])
    grad = -(X[i : i + 1].T @ (y[i : i + 1] - out))
    return grad


def forward(theta, X):
    return sigmoid(X @ theta)


def loss(theta, X, y):
    out = forward(theta, X)
    loss_i = -((y * np.log(out) + (1 - y) * np.log(1 - out)).mean())
    return loss_i


def train(theta, X, y, num_epoch, learning_rate, V_X, V_y):
    train_loss = []
    val_loss = []

    for epoch in tqdm(range(num_epoch)):
        for idx in range(len(X)):
            grad = dJ(theta, X, y, idx)
            theta -= learning_rate * grad

        train_loss.append(loss(theta, X, y))
        val_loss.append(loss(theta, V_X, V_y))

    return theta, train_loss, val_loss


def predict(theta, X):
    return forward(theta, X) > 0.5


def compute_error(y_pred, y):
    error = (y_pred != y).sum() / len(y_pred)
    return error


def write_predictions(predictions, out_file):
    with open(out_file, "w") as f:
        for pred in predictions:
            label = 1 if pred else 0
            f.write(f"{label}\n")


def write_metrics(train_error, test_error, train_loss, test_loss, out_file):
    with open(out_file, "w") as f:
        f.write(f"error(train): {train_error:.06f}\n")
        f.write(f"error(test): {test_error:.06f}\n")
        f.write(f"loss(train): {train_loss:.06f}\n")
        f.write(f"loss(test): {test_loss:.06f}\n")


def load_data(file_path):
    return np.genfromtxt(file_path, delimiter="\t", dtype="str")


def parse_data(data):
    labels = data[:, 0]
    data = np.hstack((np.ones((data.shape[0], 1)), data[:, 1:]))
    return labels, data


def main(
    formatted_train_input,
    formatted_validation_input,
    formatted_test_input,
    train_out,
    test_out,
    metrics_out,
    num_epoch,
    learning_rate,
):
    train_data = load_data(formatted_train_input).astype(float)
    validation_data = load_data(formatted_validation_input).astype(float)
    test_data = load_data(formatted_test_input).astype(float)

    train_labels, train_data = parse_data(train_data)
    validation_labels, validation_data = parse_data(validation_data)
    test_labels, test_data = parse_data(test_data)

    theta = np.zeros(train_data.shape[1])
    theta, train_loss, val_loss = train(
        theta,
        train_data,
        train_labels,
        num_epoch,
        learning_rate,
        validation_data,
        validation_labels,
    )
    train_predict = predict(theta, train_data)
    test_predict = predict(theta, test_data)

    write_predictions(train_predict, train_out)
    write_predictions(test_predict, test_out)

    train_loss = loss(theta, train_data, train_labels)
    test_loss = loss(theta, test_data, test_labels)
    train_error = compute_error(train_predict, train_labels)
    test_error = compute_error(test_predict, test_labels)
    write_metrics(train_error, test_error, train_loss, test_loss, metrics_out)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_data", type=os.path.abspath)
    parser.add_argument("--val_data", type=os.path.abspath)
    parser.add_argument("--test_data", type=os.path.abspath)
    parser.add_argument("--train_out", type=os.path.abspath)
    parser.add_argument("--test_out", type=os.path.abspath)
    parser.add_argument("--metrics_out", type=os.path.abspath)
    parser.add_argument("--num_epoch", type=int)
    parser.add_argument("--learning_rate", type=float)
    args = parser.parse_args()

    main(
        args.train_data,
        args.val_data,
        args.test_data,
        args.train_out,
        args.test_out,
        args.metrics_out,
        args.num_epoch,
        args.learning_rate,
    )
