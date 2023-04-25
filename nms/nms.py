import numpy as np


def get_area(x1, x2, y1, y2):
    return np.maximum(x2 - x1 + 1, 0) * np.maximum(y2 - y1 + 1, 0)


def nms(boxes, scores, threshold=0.5):
    # DONE
    sorted_idx = scores.argsort()[::-1]
    scores = scores[sorted_idx]
    boxes = boxes[sorted_idx]

    box_y1 = boxes[:, 0]
    box_x1 = boxes[:, 1]
    box_y2 = boxes[:, 2]
    box_x2 = boxes[:, 3]

    areas = get_area(box_x1, box_x2, box_y1, box_y2)
    order = np.arange(len(boxes))
    keep = []

    while len(order) > 0:
        keep.append(order[0])

        int_y1 = np.maximum(boxes[order[0], 0], boxes[order[1:], 0])
        int_x1 = np.maximum(boxes[order[0], 1], boxes[order[1:], 1])
        int_y2 = np.minimum(boxes[order[0], 2], boxes[order[1:], 2])
        int_x2 = np.minimum(boxes[order[0], 3], boxes[order[1:], 3])
        int_area = get_area(int_x1, int_x2, int_y1, int_y2)

        iou = int_area / (areas[order[0]] + areas[order[1:]] - int_area)

        indices = np.where(iou <= threshold)[0]
        order = order[indices + 1]

    return boxes[keep], scores[keep]


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
