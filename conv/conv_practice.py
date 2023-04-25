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

        # TODO
        self.weight = None
        self.bias = None

    def forward(self, X):
        """
        X is of shape BxC_inxHxW
        """
        # TODO
        out = None
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
