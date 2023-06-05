from argparse import ArgumentParser

import numpy as np
import cv2


def bilinear(orig_coords, image):
    fh = orig_coords[0]
    fw = orig_coords[1]
    top_h = int(fh)
    bottom_h = int(fh + 1)
    left_w = int(fw)
    right_w = int(fw + 1)
    H, W, C = image.shape

    final_color = np.zeros((3))
    if top_h >= 0 and bottom_h < H and left_w >= 0 and right_w < W:
        top_w_color = (fw - left_w) * image[top_h, right_w] + (right_w - fw) * image[
            top_h, left_w
        ]
        bottom_w_color = (fw - left_w) * image[bottom_h, right_w] + (
            right_w - fw
        ) * image[bottom_h, left_w]
        final_color = (fh - top_h) * bottom_w_color + (bottom_h - fh) * top_w_color
    return final_color


def rotate(image, angle):
    rad = np.deg2rad(angle)
    cos = np.cos(rad)
    sin = np.sin(rad)

    # rotate clockwise by angle
    R = np.array([[cos, sin], [-sin, cos]])

    H, W, C = image.shape
    orig_center = np.array([[H // 2], [W // 2]])  # 2x1

    new_H = int(W * sin + H * cos)
    new_W = int(W * cos + H * sin)
    new_center = np.array([[new_H // 2], [new_W // 2]])  # 2x1

    new_image = np.zeros((new_H, new_W, C))
    for i in range(new_H):
        for j in range(new_W):
            new_coords = np.array([[i], [j]])

            # find coordinate wrt center
            new_coords_cen = new_coords - new_center

            # invert coordinate to original space
            coords = R.T @ new_coords_cen

            # find coords in original image
            orig_coords = coords + orig_center

            new_image[i, j] = bilinear(orig_coords, image)

    return new_image


def main(args):
    image = cv2.imread("test.jpeg")
    new_image = rotate(image, args.angle)
    cv2.imwrite("rotated.jpeg", new_image)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--angle", type=float, help="angles in degrees")
    args = parser.parse_args()

    main(args)
