import numpy as np


def patchify_bf(img, patch_size):
    patches = []
    C, H, W = img.shape
    patch_H, patch_W = patch_size
    pad_H = patch_H - H % patch_H if H % patch_H != 0 else 0
    pad_W = patch_W - W % patch_W if W % patch_W != 0 else 0

    img = np.pad(img, ((0, 0), (0, pad_H), (0, pad_W)))

    for i in range(0, H, patch_H):
        for j in range(0, W, patch_W):
            patches.append(img[:, i : i + patch_H, j : j + patch_W])

    return patches


def patchify_opt(img, patch_size):
    C, H, W = img.shape
    patch_H, patch_W = patch_size
    pad_H = patch_H - H % patch_H if H % patch_H != 0 else 0
    pad_W = patch_W - W % patch_W if W % patch_W != 0 else 0
    new_H = H + pad_H
    new_W = W + pad_W

    img = np.pad(img, ((0, 0), (0, pad_H), (0, pad_W)))

    st_C, st_H, st_W = img.strides
    patches = np.lib.stride_tricks.as_strided(
        img,
        (new_H // patch_H, new_W // patch_W, C, patch_H, patch_W),
        (st_H * patch_H, st_W * patch_W, st_C, st_H, st_W),
    )

    return patches.reshape(-1, C, patch_H, patch_W)


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
