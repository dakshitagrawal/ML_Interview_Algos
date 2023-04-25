import numpy as np


def get_gradient_error(W, X, y):
    # TODO
    gradient = None
    error = None
    return gradient, error


def train(train_X, train_y):
    lr = 0.5
    np.random.seed(1)
    # TODO
    W = None

    while True:
        gradient, error = get_gradient_error(W, train_X, train_y)
        # TODO
        new_W = None

        if np.abs(new_W - W).sum() < 1e-5:
            break

        W = new_W

    print(f"Train error: {error}")
    print(f"Weights: {W}")
    return W


def main():
    data_x = np.linspace(1.0, 10.0, 100)[:, None]
    np.random.seed(1)
    data_y = np.sin(data_x) + 0.1 * data_x ** 2 + 0.5 * np.random.randn(100, 1)
    data_x /= data_x.max()

    # Fold in bias
    data_x = np.hstack((np.ones_like(data_x), data_x))

    # Split data to train/test
    np.random.seed(1)
    shuffle = np.random.permutation(len(data_x))
    split = 20
    test_x = data_x[shuffle[:split]]
    test_y = data_y[shuffle[:split]]
    train_x = data_x[shuffle[split:]]
    train_y = data_y[shuffle[split:]]

    W = train(train_x, train_y)
    _, test_error = get_gradient_error(W, test_x, test_y)
    print(f"Test error: {test_error}")


if __name__ == "__main__":
    main()

    # Final output should be:
    # Train error: 0.6682580089505128
    # Weights: [[-1.98345207]
    # [10.86604795]]
    # Test error: 0.88779183536169
