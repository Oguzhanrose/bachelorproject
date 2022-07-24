import torch
from checker import assert_types
from collections import Counter
from bounding_boxes import cell2corner
from general_utils import non_max_suppression, intersection_over_union
import numpy as np


def mean_batch_map(prediction_cell:torch.Tensor, labels_cell:torch.Tensor, config):
    """
    Calculate the mean average precision (mAP) between `prediction_cell` and `label_cell`.
    Both are expected to be batches of bounding boxes in YOLO cell-format format:
        torch.tensor([
            S*S*[p_object, x_cell, y_cell, w_cell, h_cell, C_1, C_2 ..., C_n],
            S*S*[p_object, x_cell, y_cell, w_cell, h_cell, C_1, C_2 ..., C_n],
            ...
        ])

    :param prediction_cell: A batch of predicted labels in yolo-format
    :param labels_cell: A to `prediction_cell` corresponding batch of ground truth labels in yolo-format
    :param config: A class containing configuration information. The following 7 attributes most be defined within this:
                   (1) image_size:int
                   (2) C:int --> number of possible classes a BB can take
                   (3) nms_conf_threshold:int
                   (4) nms_iou_threshold:int
                   (5) class_sensitive:bool
                   (6) map_iou_thresholds: list --> e.g. [0.5, 0.55, ... 0.95])
                   (7) apply_nms_before_map: bool
    :param apply_nms: If True, will apply Non-max-suppression with the hyperparameters defined in `config`
    :return:  dict with mAP-scores for every threshold defined in `config.map_iou_thresholds` + their combined average
    """
    # Checks
    assert_types([prediction_cell, labels_cell], [torch.Tensor, torch.Tensor])

    # Helper
    def _prepare_for_map(bbs_images: list) -> list:
        clean = []
        for image_index, bbs in enumerate(bbs_images):
            for bb in bbs:
                clean.append([image_index, bb[5], bb[0]] + bb[1:5])
        return clean

    # Preapre predictions
    prediction_cell = prediction_cell.clone()
    if config.apply_nms_before_map:
        prediction_cell[..., 3:5] = torch.abs(prediction_cell[..., 3:5])  # Ensure width and height cannot take negative values
        prediction_corner = []
        for out_cell in prediction_cell:
            nms_image = non_max_suppression(out_cell, config.image_size, config.C, config.nms_conf_threshold, config.nms_iou_threshold, config.class_sensitive)
            prediction_corner.append(nms_image)
    else:
        prediction_corner = [cell2corner(pred, config.image_size).tolist() for pred in prediction_cell]


    # Prepare labels
    labels_cell = labels_cell.clone()
    labels_corner = [cell2corner(label, config.image_size).tolist() for label in labels_cell]

    # Average mAP over the entire batch and all the thresholds
    map_scores = {}
    for map_iou_threshold in config.map_iou_thresholds:
        map_score = _mean_average_precision(
            _prepare_for_map(prediction_corner),
            _prepare_for_map(labels_corner),
            map_iou_threshold,
            config.C
        )
        map_scores[map_iou_threshold] = map_score
    map_scores["average_over_all_thresholds"] = np.mean(list(map_scores.values()))
    return map_scores


def _mean_average_precision(pred_boxes:list, true_boxes:list, iou_threshold:float, num_classes:int) -> float:
    """
        Calculates mean average precision between `pred_boxes` and `true_boxes`.
    Both should be in the format:
        [
            [train_idx, class_prediction, prob_score, x1, y1, x2, y2],
            [train_idx, class_prediction, prob_score, x1, y1, x2, y2],
            ...
        ]

    NOTE:
        The code is copied (insignificant modifications has been made, but all credit goes to him) from:
        https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/metrics

    :param pred_boxes: Predicted labels, format: [train_idx, class_prediction, prob_score, x1, y1, x2, y2],
    :param true_boxes: Ground truth labels, format: [train_idx, class_prediction, prob_score, x1, y1, x2, y2],
    :param iou_threshold: IoU threshold which determine if a BB is correct or not
    :param num_classes: Number of possible classes
    :return: mAP value across all classes given a specific IoU threshold
    """
    # Checks
    assert_types([pred_boxes, true_boxes, iou_threshold, num_classes], [list, list, float, int])
    assert 0 < iou_threshold < 1.0, "IoU threshold must be in range 0-1"

    # list storing all AP for respective classes
    average_precisions = []

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets, and only add the ones that belong to the current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example. Counter here finds how many ground truth bboxes we get for each training example, so let's say img 0 has 3, img 1 has 5 then we will obtain a dictionary with: amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary and convert to the following (w.r.t same example): ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same training idx as detection
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):

                iou = intersection_over_union(
                    torch.tensor(detection[3:]).unsqueeze(0),
                    torch.tensor(gt[3:]).unsqueeze(0)
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + 1e-12)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-12)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # plt.plot(recalls.tolist(), precisions.tolist(),"o-", label=dataset.class_int_to_name[c])

        # Area under det PR-curve
        average_precisions.append(
            torch.trapz(precisions, recalls) # torch.trapz is for numerical integration
        )

    return float(sum(average_precisions) / len(average_precisions))
