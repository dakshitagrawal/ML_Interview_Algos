import numpy as np
import torch
import torch.nn.functional as f


class Conv:
    def __init__(self, c_in, c_out, kernel, stride=1, padding=0, dilation=1):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.c_in = c_in
        self.c_out = c_out

        self.weight = np.random.rand(self.c_out, self.c_in, self.kernel, self.kernel)
        self.bias = np.random.rand(self.c_out)

    def forward(self, X):
        """
        X is of shape BxC_inxHxW
        """

        B, cx_in, H, W = X.shape
        assert cx_in == self.c_in

        H_out = (
            H + 2 * self.padding - ((self.kernel - 1) * self.dilation + 1)
        ) // self.stride + 1
        W_out = (
            W + 2 * self.padding - ((self.kernel - 1) * self.dilation + 1)
        ) // self.stride + 1

        X = np.pad(
            X,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
        )
        out = np.zeros((B, self.c_out, H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                w2 = (self.kernel // 2) * self.dilation
                in_i = w2 + i * self.stride
                in_j = w2 + j * self.stride
                patch = X[
                    :,
                    :,
                    in_i - w2 : in_i + w2 + 1 : self.dilation,
                    in_j - w2 : in_j + w2 + 1 : self.dilation,
                ]
                out[:, :, i, j] = (patch[:, None] * self.weight[None]).sum(
                    (2, 3, 4)
                ) + self.bias[None]

        return out

    def forward_torch(self, X):
        out = f.conv2d(
            torch.from_numpy(X),
            torch.from_numpy(self.weight),
            torch.from_numpy(self.bias),
            self.stride,
            self.padding,
            self.dilation,
        )
        return out.numpy()


def main():
    img = np.random.randn(1, 3, 10, 10)
    conv = Conv(3, 2, 3, 2, 1, 2)
    out2 = conv.forward_torch(img)
    out1 = conv.forward(img)
    assert (out1 - out2).sum() < 1e-10


if __name__ == "__main__":
    main()
