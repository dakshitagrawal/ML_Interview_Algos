import numpy as np


def patchify_bf(img, patch_size):
    # TODO
    patches = None
    return patches


def patchify_opt(img, patch_size):
    # TODO
    patches = None
    return patches


def main():
    H = 10
    W = 12
    patch_size = (5, 5)
    img = np.arange(H * W * 1).reshape(1, H, W)
    print(img)

    patches_bf = patchify_bf(img, patch_size)
    print(patches_bf)

    patches_opt = patchify_opt(img, patch_size)
    assert (patches_opt == patches_bf).all()


if __name__ == "__main__":
    main()
