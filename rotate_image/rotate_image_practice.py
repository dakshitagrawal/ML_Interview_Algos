from argparse import ArgumentParser

import numpy as np
import cv2


def rotate(image, angle):
    # TODO
    new_image = None
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
