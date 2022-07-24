import warnings
import checker
import torch
from typing import List
import os
import numpy as np
import math


def cell2corner(boxes:torch.Tensor, image_size:int, B:int=1, onehot_class:bool=False) -> torch.Tensor:

    """
    Convert `boxes` from
        S*S*[p_object, x_cell, y_cell, w_cell, h_cell, C_1, C_2 ..., C_n]
    to
        torch.tensor([
            [p_object, x1_corner, y1_corner, x2_corner, y2_corner, C_0, C_1, ... C_n],
            [p_object, x1_corner, y1_corner, x2_corner, y2_corner, C_0, C_1, ... C_n],
            ...
        ])
    if `onehot_class` is True. If it's False:
        torch.tensor([
            [p_object, x1_corner, y1_corner, x2_corner, y2_corner, class_label_int],
            [p_object, x1_corner, y1_corner, x2_corner, y2_corner, class_label_int],
            ...
        ])

    :param boxes: Bounding boxes in cell format
    :param image_size: size of the image the bounding boxes belong to
    :param B: Number of bounding boxes within each cell
    :param onehot_class: If true, will return classes in ont hot format. If false, will return
    :return: torch.Tensor with corner bounding box annotations (i.e. normal cartesian coordinates), shape --> S,S,5*B+C
    """

    # Checks
    checker.assert_types([boxes, image_size, B, onehot_class], [torch.Tensor, int, int, bool])

    # Setup
    S = boxes.shape[0]
    cell_size = int(image_size / S)
    boxes = boxes.clone().detach().cpu()

    if (image_size / S) % 1 != 0: raise ValueError("img_size must be an divisible by 2")
    if len(boxes.shape) != 3: raise ValueError("Shape mismatch, did you to pass a batch instead?")

    # Convert bb from cell to corner coordinates
    to_return = []
    for row in range(S):
        for col in range(S):
            if not boxes[row, col, :].any():
                continue

            p_obj = float(boxes[row, col, 0])
            x, y = boxes[row, col, 1], boxes[row, col, 2]
            w_half = boxes[row, col, 3] / 2
            h_half = boxes[row, col, 4] / 2

            x1 =  float((x - w_half + row) * cell_size)
            y1 =  float((y - h_half + col) * cell_size)
            x2 =  float((x + w_half + row) * cell_size)
            y2 =  float((y + h_half + col) * cell_size)

            if not onehot_class:
                to_return.append([p_obj, x1, y1, x2, y2, boxes[row, col, 5:].argmax()])
            else:
                to_return.append([p_obj, x1, y1, x2, y2] + boxes[row, col, 5:].tolist())
    return torch.Tensor(to_return)


# TODO: refactor
def yolo2cell(boxes:torch.Tensor, img_size:int, S:int, C:int, B:int=1) -> torch.Tensor:
    """
    Convert `boxes` from yolo-format
        [
        [class_label, x_center, y_center, w, h],
        [class_label, x_center, y_center, w, h],
        ...
        ]
    to a torch tensor of shape:
        S*S*[p_object, x_cell, y_cell, w_cell, h_cell, C_1, C_2 ..., C_n]

    :param boxes: Bounding box annotations in yolo-format
    :param img_size: The size of the image
    :param S: Number of cells used to encode the bounding boxes
    :param B: Number of bounding boxes within each cell
    :param C: Number of classes used for one hot encoding
    :return: torch.Tensor with cell relative bounding box annotations, shape --> S,S,5*B+C
    """

    # Checks
    checker.assert_types([boxes, S, B], [torch.Tensor, int, int])
    if (len(boxes) == 0) or (img_size < 32) or (S < 1) or (B not in [1, 2]):
        raise ValueError(f"One or more of the inputs are invalid - values respectively `{len(boxes), img_size, S, B}`")

    # Setup
    boxes[:, 1:] = torch.clip(boxes[:, 1:], min=0.0, max=0.99999)
    label_matrix = torch.zeros((S, S, 5 * B + C))
    grid_cell = img_size / S

    # Cell relative encoding
    for box in boxes:
        class_label = box[0]
        x_pixel, y_pixel = box[1] * img_size, box[2] * img_size
        x_cell, x_index = math.modf(x_pixel / grid_cell)
        y_cell, y_index = math.modf(y_pixel / grid_cell)
        x_index = int(x_index)
        y_index = int(y_index)
        box[1] = x_cell
        box[2] = y_cell

        w, h = box[3], box[4]
        w_cell, h_cell = w * img_size / grid_cell, h * img_size / grid_cell
        box[3] = w_cell
        box[4] = h_cell

        one_hot = [0 for _ in range(C)]
        one_hot[int(class_label.detach())] = 1

        # Make sure that there's not 2 objects within the same grid cell
        if any(label_matrix[x_index, y_index, :] != 0):
            warnings.warn("Detected more then one object within the same cell. This object will be skipped."
                          "Consider increasing `S` to avoid this.")
            continue

        label_matrix[x_index, y_index, :] = torch.tensor([1] + box[1:5].detach().tolist() + one_hot)


    return label_matrix


def read_yolo_from_file(label_path:str) -> List[list]:
    """
    Load yolo label from file. Expect each line to be in the format format:
        class_label x_center y_center bb_width bb_height.


    :param label_path: Takes a label_path, which is a string
    :return: list of lists with BBs in the format:
            [
            [class_label, x_center, y_center, bb_width, bb_height],
            [class_label, x_center, y_center, bb_width, bb_height],
            ...
            ]
    """
    # Checks
    checker.assert_type(label_path, str)
    if not os.path.exists(label_path): raise ValueError("Received a bad path")
    if label_path[-4:] != ".txt": raise ValueError(f"Expected .txt file but received {os.path.basename(label_path)}")

    # Read labels
    boxes = []
    with open(label_path) as f:
        for label in f.readlines():

            # Extract label info + label checks
            label_split = label.replace("\n", "").split(" ")
            if len(label_split) != 5:
                raise RuntimeError("Received a bad label")
            class_label, x, y, w, h = [float(x) for x in label_split]
            if not all(0<=number<=1 for number in [x,y,w,h]):
                raise RuntimeError("one or more of [x,y,w,h] is outside the accepted range [0,1]")

            boxes.append([class_label, x, y, w, h])

    return boxes


if __name__ == "__main__":
    # annotation_paths = natsorted([name for name in glob(label_path)])
    path = "C:/Users/JK/Desktop/Bachelor_annoteringsdata/EGMONT_11-12-2021/egmont_11-12-2021_file0025_small_reduced/obj_train_data/frame_000017.txt"
    yolo_labels = read_yolo_from_file(path)
    cell_label = yolo2cell(yolo_labels, 512, 2, 2)
    np.set_printoptions(suppress=True)


