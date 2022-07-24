import albumentations as A
from albumentations.pytorch import ToTensorV2
import os.path
import dutils as U
import numpy as np
import torch
import cv2
from dutils.jupyter_ipython import show_image as show

def draw_bbs(image: np.ndarray, labels: torch.Tensor, label_map: dict, confs: list = None, show_image: bool = True):
    """
    Draw BB, labels and confidence-scores onto an image.

    :param image: Image which is used to draw on
    :param labels: standard YOLO-annotations [class_label, x, y, w, h]
    :param label_map: A dict of names corresponding to each possible label e.g. { 0:(10,10,10}, 1: ...}
    :param confs: A list of confidence scores given as a float (Same length as `anno`)
    :param show_image: Show image in place instead of returning it
    :return: None or drawn upon
    """

    if "int" not in str(labels.dtype):
        raise ValueError(f"Expect `labels` to be of type int, but received {labels.dtype}."
                         f"If you forgot to cast it to an int try `labels.long()`")


    class_labels = labels[:, 0].tolist()
    bbs = labels[:, 1:].tolist()
    unique_labels = list(label_map.keys())


    # Pick the same number of predefined/random colors as there are unique labels
    predefined_colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
                         (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]
    color_map = {label: color for label, color in zip(unique_labels, predefined_colors)}

    if len(unique_labels) > 10:
        for label in unique_labels[10:]:
            color_map[label] = U.colors.random_color(1, "rgb", 50, 150)

    image = image.copy()
    confs = [None] * len(bbs) if confs is None else confs
    for (bb, class_label, conf) in zip(bbs, class_labels, confs):
        image = draw_bb(image, bb, label_map[class_label], conf, color_map[class_label])

    if show_image:
        U.images.show_ndarray_image(image, BGR2RGB=False)
    else:
        return image


def draw_bb(image: np.ndarray, bb: list, label: str = None, conf: float = None, color: tuple = None) -> np.ndarray:
    """
    Add `bb` to `image` with optional text `label` and `conf` score.

    :param image: image in np.ndarray format
    :param bb: Bounding box in cartesian-format i.e [x_corner, y_corner, width_corner, height_corner]
    :param color: Color of the drawn `bb`. If `None` color gets assigned randomly
    :param label: The predicted class-label of `bb`
    :param conf: Confidence score for the bb's `label`
    """

    # Setup
    image = image.copy()
    color = color if (color is not None) else U.colors.random_color(1, "rgb", 50, 150)
    p1, p2 = (bb[0], bb[1]), (bb[2], bb[3])
    if p1[0] > p2[0]: p1, p2 = p2, p1  # Otherwise the plot get messed up
    font = cv2.FONT_HERSHEY_DUPLEX

    # Determine font_size and font_thickness as a function of the image height (found experimentally)
    font_size = max(0.5, 0.5 * (image.shape[0] // 512))
    font_thickness = max(1, int(np.floor(font_size)))

    # Draw Bounding box
    cv2.rectangle(image, p1, p2, color, 2)

    # Add optional label and confidence score
    if label or conf:
        text = label + (": " if conf else "")
        text = str(round(conf * 100, 3)) + "% " + text if conf else text

        (w, h), _ = cv2.getTextSize(text, font, font_size, font_thickness)
        cv2.rectangle(image, (p1[0], p1[1]), (p1[0] + w, p1[1] - h), color, -1, cv2.LINE_AA)
        cv2.putText(image, text, (p1[0], p1[1] - 2), font, font_size, (200, 200, 200), font_thickness, cv2.LINE_AA)

    return image
