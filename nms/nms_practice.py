import numpy as np


def nms(boxes, scores, threshold=0.5):
    # TODO
    nms_boxes = None
    nms_scores = None
    return nms_boxes, nms_scores


def main():
    """
    Boxes are [y1, x1, y2, x2]
    """
    boxes = np.array([(187, 82, 337, 317), (150, 67, 305, 282), (246, 121, 368, 304)])
    scores = np.array([0.9, 0.75, 0.8])
    nms_boxes, nms_scores = nms(boxes, scores)
    assert (nms_boxes == np.array([(187, 82, 337, 317), (246, 121, 368, 304)])).all()
    assert (nms_scores == np.array([0.9, 0.8])).all()

    boxes = np.array(
        [
            [220, 220, 420, 420],
            [200, 200, 400, 400],
            [1, 1, 2, 2],
            [200, 240, 400, 440],
            [240, 200, 440, 400],
        ]
    )
    scores = np.array([0.8, 0.9, 0.5, 0.7, 0.6])
    nms_boxes, nms_scores = nms(boxes, scores)
    assert (nms_boxes == np.array([(200, 200, 400, 400), (1, 1, 2, 2)])).all()
    assert (nms_scores == np.array([0.9, 0.5])).all()


if __name__ == "__main__":
    main()
