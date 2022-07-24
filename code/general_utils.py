from checker import assert_types
from bounding_boxes import cell2corner
import torch
import random
import os
import numpy as np


def intersection_over_union(boxes_preds: torch.Tensor, boxes_targets: torch.Tensor) -> torch.Tensor:
    """
    Calculates IoU between `boxes_preds` and `boxes_targets`. The BBs should be tensors in cartesian format:
        torch.Tensor([
            [x1, y1, x2, y2],
            [x1, y1, x2, y2],
            ...
        ])

    :param boxes_preds: BB in cartesian format i.e. [x1, y1, x2, y2]
    :param boxes_targets: BB in cartesian format i.e. [x1, y1, x2, y2]
    :param generalized_iou: An alternative approach to IoU which takes distance into account (NYI)
    :return: A tensor of floats between 0 to 1.
             This encode a percentage of how much `boxes_preds` are overlapping with the corresponding `box_targets`.
    """

    # Checks
    assert_types([boxes_preds, boxes_targets], [torch.Tensor, torch.Tensor])
    assert len(boxes_preds.shape) == 2 and len(boxes_targets.shape) == 2, "Bad shape"
    assert boxes_preds.shape[1] == 4 and boxes_targets.shape[1] == 4, "Bad shape"

    # Prediction box - Upper left corner intersection
    predicted_box_x1 = boxes_preds[..., 0]
    predicted_box_y1 = boxes_preds[..., 1]
    predicted_box_x2 = boxes_preds[..., 2]
    predicted_box_y2 = boxes_preds[..., 3]

    target_box_x1 = boxes_targets[..., 0]
    target_box_y1 = boxes_targets[..., 1]
    target_box_x2 = boxes_targets[..., 2]
    target_box_y2 = boxes_targets[..., 3]

    # Calculates intersection between the predicted bbs and the ground truth bb
    x1 = torch.max(predicted_box_x1, target_box_x1)
    x2 = torch.min(predicted_box_x2, target_box_x2)
    y1 = torch.max(predicted_box_y1, target_box_y1)
    y2 = torch.min(predicted_box_y2, target_box_y2)

    # Calculate box areas
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    predicted_box_area = abs((predicted_box_x2 - predicted_box_x1) * (predicted_box_y2 - predicted_box_y1))
    target_box_area = abs((target_box_x2 - target_box_x1) * (target_box_y2 - target_box_y1))

    # Intersection over union
    iou_score = intersection / (predicted_box_area + target_box_area - intersection + 1e-9)
    return iou_score.flatten()



def seed_torch(seed:int, deterministic:bool = False):
    torch.backends.cudnn.deterministic = deterministic
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def adjust_bb(label:list, image_size:int):
    """

    Adjust the width and height of the bounding box within `label` by 1 pixel if it's to close to the edge.
    This prevent rounding errors which will throw an error do to rounding imprecision in the Albumentations augmentation

    :param label: label in yolo-format i.e. [class_label, x_center, y_center, w, h]
    :param image_size: image size of the image label corresponds to
    :return: `label` (still in yolo-format) that has a been cropped if necessary
    """

    _, x, y, w, h = label
    pixel_limit = 1 / image_size

    # crop x-axis
    if (x - w/2) < pixel_limit:
        label[3] = w - pixel_limit
    elif (x + w/2) > (1 - pixel_limit):
        label[3] = w - pixel_limit

    # crop y-axis
    if (y - h/2) < pixel_limit:
        label[4] = h - pixel_limit
    elif (y + h/2) > (1 - pixel_limit):
        label[4] = h - pixel_limit

    return label


def _nms_helpers(preds_BB:list, conf_threshold:float, IOU_threshold:float) -> list:
    """
    Remove overlapping and low confidence bounding boxes. Input is in corner format:
    [
     [P_obj, x1, y1, x2, y2, C1, C2, ..., Cn],
     [P_obj, x1, y1, x2, y2, C1, C2, ..., Cn],
     ....
    ]
    :param preds_BB: Pytorch label tensor in corner format
    :param conf_threshold: All bounding boxes with a confidence score less the this value will be removed initially
    :param IOU_threshold: The maximum IoU overlap that will be accepted during the NMS algorithm.
    :return: Remaining bounding boxes with conf_score < Threshold and no IoU_overlap > IOU_threshold
    """

    # Checks
    assert_types([preds_BB, conf_threshold, IOU_threshold], [list, float, float])
    if len(preds_BB) == 0:
        return []

    remaining_BB = []

    preds_BB_to_sort = [BB for BB in preds_BB if BB[0] >= conf_threshold]
    preds_BB_to_sort.sort(reverse = True, key=lambda x: x[0])

    while len(preds_BB_to_sort) > 0:
        current_BB = preds_BB_to_sort.pop(0)
        remaining_BB.append(current_BB)
        for BB in preds_BB_to_sort:

            # IoU
            iou_score = intersection_over_union(
                torch.tensor([current_BB[1:5]]),
                torch.tensor([BB[1:5]])
            )

            if iou_score > IOU_threshold:
                preds_BB_to_sort.remove(BB)

    return remaining_BB


def non_max_suppression(preds_cell:torch.Tensor, image_size:int, C:int, conf_threshold:float=0.5, iou_threshold:float=0.75, class_sensitive:bool=True):
    """ Takes a single batch of SxSx(5+C) cell format"""

    # Checks
    assert_types([preds_cell, image_size, conf_threshold, iou_threshold], [torch.Tensor, int, float, float])
    assert (0<conf_threshold<1) and (0<iou_threshold<1), "`iou_threshold` and `conf_threshold` most be in range (0,1)"
    if len(preds_cell.shape) != 3: raise ValueError("Shape mismatch, did you pass a batch? If so, try boxes[0]")

    bbs_after_nms = []

    # Ignore classes
    if not class_sensitive:
        preds_corner = cell2corner(preds_cell, image_size, onehot_class=False)
        bbs_after_nms = _nms_helpers(preds_corner.tolist(), conf_threshold, iou_threshold)
        return bbs_after_nms

    # Class sensitive
    preds_corner = cell2corner(preds_cell, image_size, onehot_class=True)
    predicted_classes = preds_corner[:, 5:].argmax(-1)
    for class_index in range(C):
        class_preds_corner = preds_corner[predicted_classes == class_index].tolist()
        bbs_after_nms += _nms_helpers(class_preds_corner, conf_threshold, iou_threshold)
        raise RuntimeError("Multiclass NMS has not been check at all and should probably not be used as is!")

    return bbs_after_nms



#print(adjust_bb([0, 0.594032, 0.5,      0.811936, 1.      ], 400))
#print(adjust_bb([0, 0.566825, 0.365056, 0.484198, 0.730113], 400))
