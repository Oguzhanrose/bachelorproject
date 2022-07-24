import albumentations as A
from albumentations.pytorch import ToTensorV2
from checker import assert_types
from typing import List
from glob import glob
import numpy as np
import torch
import cv2
import os
import dutils as U
import checker

# Albumentation - settings for working with bounding boxes
_BB_PARAMS = A.BboxParams(
    format='yolo',
    min_area=10,
    min_visibility=0.1,
    label_fields=['class_labels']
)


def _get_shared_augmentation (width:int, height:int) -> list:
    # Found partial through trial and error and partial through https://albumentations-demo.herokuapp.com/
    assert_types([width, height], [int, int])
    return [
        A.RandomSizedBBoxSafeCrop(width=width, height=height, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.2),
        A.HueSaturationValue(20, 40, 50, p=0.25),
        A.MotionBlur(blur_limit=15, p=1.),
        A.GaussNoise(var_limit=(10, 50), p=0.25),
        A.Cutout(num_holes=10, max_h_size=20, max_w_size=20, p=0.25),
        A.ToGray(p=0.1)
    ]


class BasicAugmentation:
    """
    Image augmentation class capable of applying basic augmentations found within the albumentations library.

    NOTE:
    This class is supposed to be instantiated as a function which can used through __call__ e.g.
    >> basic = BasicAugmentation(512)
    >> augmented_image, augmented_label = basic(image, label)
    The only reason it's a class in the first place is because it was more convenient for bundling everything together
    """
    def __init__(self, img_size:int, norm_means:tuple=(0,0,0), norm_stds:tuple=(1,1,1)):
        """
        :param img_size: image size post augmentation
        :param norm_means: Normalization means for each color channel
        :param norm_means: Normalization standard deviation for each color channel
        """

        assert (len(norm_means) == 3) and (len(norm_stds) == 3), "Received bad normalization constant(s)"
        self.transform_normal = A.Compose([
            *_get_shared_augmentation(img_size, img_size),
            A.Normalize(mean=norm_means, std=norm_stds),
            ToTensorV2()
        ], bbox_params=_BB_PARAMS)


    def __call__(self, image:np.ndarray, label:np.ndarray) -> (torch.Tensor, torch.Tensor):
        """
        Apply basic augmentations to `image`.

        :param image: color image in numpy format
        :param label: bounding box annotations in yolo format i.e. [class_label, x_center, y_center, w, h]
        :return: squared augmented color torch image and its corresponding label as torch.tensor
        """
        assert_types([image, label], [np.ndarray, np.ndarray])

        # Perform augmentation on both image and bb
        transformed = self.transform_normal(image=image, bboxes=label[:, 1:], class_labels=label[:, 0])

        # Extract the augmented image and its augmented bb
        transformed_image = transformed["image"]
        to_merge = np.array(transformed["class_labels"]).reshape(-1, 1), np.array(transformed["bboxes"])
        transformed_label = np.concatenate(to_merge, 1)

        return transformed_image, torch.tensor(transformed_label)


class InferenceAugmentation:
    """
    Augmentations intended for inference: Resize, normalization and ToTensor

    NOTE:
    This class is supposed to be instantiated as a function which can used through __call__ e.g.
    >> inference_augment = InferenceAugmentation(512)
    >> augmented_image, none_label = inference_augment(image, None)
    The only reason it's a class in the first place is because it was more convenient for bundling everything together
    """

    def __init__(self, img_size:int, norm_means:tuple=(0,0,0), norm_stds:tuple=(1,1,1)):
        """
        :param img_size: image size post augmentation
        :param norm_means: Normalization means for each color channel
        :param norm_means: Normalization standard deviation for each color channel
        """
        assert (len(norm_means) == 3) and (len(norm_stds) == 3), "Received bad normalization constant(s)"
        self.image_size = img_size
        self.augmentation = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=norm_means, std=norm_stds),
            ToTensorV2()
        ])

    def __call__(self, image:np.ndarray) -> torch.Tensor:
        """
        Prepare ´image` for inference.

        :param image: color image in numpy format
        :return: squared augmented torch image
        """
        assert_types([image], [np.ndarray])
        transformed = self.augmentation(image=image)
        return transformed["image"]


class SpecialEffects:
    """
    Image augmentation class capable of applying 4 special effects designed to emulate different
    real world scenarios that partly obscure the camera lens:

    (1) droplets_and_streaks: Rain drops and rain streaks
    (2) large_droplets: Large rain drops
    (3) stains_and_streaks: emulate finger prints, dirt debris and dried rain
    (4) scratches: Light scratches

    NOTE:
    This class is supposed to be instantiated as a function which can used through __call__ e.g.
    >> special_effects = SpecialEffects("./augmentations_images/", 416, 416)
    >> augmented_image = special_effects(image)
    The only reason it's a class in the first place is because it was more convenient for bundling everything together
    """
    def __init__(self, width:int, height:int, norm_means:tuple=(0,0,0), norm_stds:tuple=(1,1,1), folder_path:str="../augmentations_images"):
        """
        :param width: special image width
        :param height: special image height
        :param norm_means: Normalization means for each color channel
        :param norm_means: Normalization standard deviation for each color channel
        :param folder_path: Folder path containing exactly four 3K special effects images (jpg)
        """

        # Initial checks
        checker.assert_types([width, height, norm_means, norm_stds, folder_path], [int, int, tuple, tuple, str])
        assert (len(norm_means) == 3) and (len(norm_stds) == 3), "Received bad normalization constant(s)"
        assert os.path.isdir(folder_path), "Received bad folder path"
        assert len(glob(os.path.join(folder_path,"*.jpg"))) == 4, \
            f"Expected folder_path to contain exactly 4 jpg-images, but received {len(folder_path)}"

        # Setup
        self.width = width
        self.height = height
        self.effect_names = ["stains_and_streaks", "droplets_and_streaks", "scratches", "large_droplets"]
        self.effect_images = {effect_name: None for effect_name in self.effect_names}


        # Augmentations
        self.transform_pre_augment = A.Compose(
            _get_shared_augmentation(width, height),
            bbox_params=_BB_PARAMS
        )
        self.transform_post_augment = A.Compose([A.Normalize(mean=norm_means, std=norm_stds), ToTensorV2()])
        self.random_crop = A.RandomCrop(height, width)

        # Loading images - setup
        image_names = [
            'LiquidStainsStreaks005_OVERLAY_VAR1_3K.jpg',
            'RainDropsAndStreaks001_COL_3K.jpg',
            'ScratchesLight004_OVERLAY_VAR2_3K.jpg',
            'WaterDropletsMixedBubbled001_COL_3K.jpg'
        ]
        paths = [os.path.join(folder_path, n) for n in image_names]

        # Loading images - load and preprocess
        for i, path in enumerate(paths):
            assert os.path.exists(
                path), "`folder_path` contain at least 1 error. Should only contain the 4 images in `image_names`"
            effect_name = self.effect_names[i]
            image = cv2.imread(path)

            # If greyscale, copy 3 times to form R, G and B. Shape change: (H, W) --> (H, W, 3)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.merge([image] * 3)

            # Histogram stretching
            image = self._histogram_stretching(image)

            # Gamma correction (It just happend to work better for scratches and large_droplets this way)
            if effect_name in ["scratches", "large_droplets"]:
                image = self._gamma_correction(image, 0.50)

            self.effect_images[effect_name] = image

    @staticmethod
    def _histogram_stretching(image:np.ndarray) -> np.ndarray:
        min_vd, max_vd = 0, 255
        min_v, max_v = np.min(image), np.max(image)
        scaling_coef = (max_vd - min_vd) / (max_v - min_v)
        stretched_image = scaling_coef * (image - min_v) + min_vd
        return stretched_image.astype(np.uint8)

    @staticmethod
    def _gamma_correction(image:np.ndarray, gamma:float) -> np.ndarray:
        table = [((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]
        table_right_format = np.array(table).astype("uint8")
        return cv2.LUT(image, table_right_format)


    def __call__(self, original_image:np.ndarray, label:np.ndarray, effect_name:str="random") -> (torch.Tensor, torch.Tensor):
        """
        Apply `effect_name` to `original_image` and adjust the bounding boxes within `label` to the cropping
        applied by the special effect.

        :param original_image: numpy color image with shape (self.height, self.width, 3)
        :param label: bounding box annotations in yolo format i.e. [class_label, x_center, y_center, w, h]
        :param effect_name: The effect that will be applied, must be in:
                             ["stains_and_streaks", "droplets_and_streaks", "scratches", "large_droplets"]
                             NOTE: if "random", will randomly (uniform) pick one of the 4 effects
        :return: numpy image with the special effect applied
        """
        assert_types([original_image, label, effect_name], [np.ndarray, np.ndarray, str])
        assert effect_name in self.effect_names+["random"], \
            f"´{effect_name}` is not a valid effect_name. Legal effect names: `{self.effect_names+['random']}`"

        # Set effect
        if effect_name == "random":
            effect_name = np.random.choice(self.effect_names)
        effect_image = self.effect_images[effect_name]

        if effect_name in ["scratches", "large_droplets"]:
            effect_image_cropped = A.RandomCrop(150, 150)(image=effect_image)["image"]
            effect_image_cropped = cv2.resize(effect_image_cropped, (self.height, self.width))
            effect_image_cropped = self._histogram_stretching(effect_image_cropped)
        else:
            effect_image_cropped = self.random_crop(image=effect_image)["image"]
        mask = effect_image_cropped / 255

        # Extract the augmented image and its augmented bb
        transformed = self.transform_pre_augment(image=original_image, bboxes=label[:, 1:], class_labels=label[:, 0])
        transformed_image = transformed["image"]
        to_merge = np.array(transformed["class_labels"]).reshape(-1, 1), np.array(transformed["bboxes"])
        transformed_label = np.concatenate(to_merge, 1)

        # Effect
        blurred_image = cv2.blur(transformed_image.copy(), (30, 30))
        image_effect = blurred_image * mask + transformed_image * (1 - mask)
        final_image = self.transform_post_augment(image=image_effect)["image"]

        return final_image, torch.tensor(transformed_label)


class MosaicAugmentation:
    def __init__(self, img_size: int, norm_means: tuple = (0, 0, 0), norm_stds: tuple = (1, 1, 1)):
        # Setup
        checker.assert_types([img_size, norm_means, norm_stds], [int, tuple, tuple])
        assert (len(norm_means) == 3) and (len(norm_stds) == 3), "Received bad normalization constant(s)"
        assert (img_size / 2) % 2 == 0, "`img_size` must be divisible by 2"

        # Setup
        self.img_size = img_size
        self.transform_pre_augment = A.Compose(
            _get_shared_augmentation(img_size//2, img_size//2),
            bbox_params=_BB_PARAMS
        )
        self.transform_post_mosaic = A.Compose([A.Normalize(mean=norm_means, std=norm_stds), ToTensorV2()])


    def __call__(self, images_in:List[np.ndarray], labels_in:List[np.ndarray], img_size:int=512) -> (torch.Tensor, torch.Tensor):
        """
        A modified version of mosaic image augmentation as implemented in YOLOv4

        NOTE:
        this code is heavily inspired (most is copied directly) by ultralytic's implementation:
        https://github.com/ultralytics/yolov3/blob/master/utils/datasets.py

        :param images_in: 4 squared (i.e. height == width) images with identical side lengths which (side lengths must be divisible by 2)
        :param labels_in: 4 numpy arrays with bounding box annotations in yolo format [x_c, y_c, w_c, h_c, C_0, C_1, ..., C_n]
        :param img_size: The size of the final image
        :return: mosaic image and its corresponding bounding boxes (both as tensors if `return_tensors` else in numpy fomrat)
        """

        # checks
        assert_types([images_in, labels_in, img_size], [list, list, int])
        assert len(images_in) == len(labels_in), "Shape mismatch between the length of `images_in` and `labels_in`"

        # Augment images and their corresponding bounding boxes
        images = []
        labels_original = []
        for (image, label) in zip(images_in, labels_in):
            # Augment
            transformed = self.transform_pre_augment(image=image,  bboxes=label[:, 1:], class_labels=label[:, 0])
            transformed_image = transformed["image"]
            to_merge = np.array(transformed["class_labels"]).reshape(-1, 1), np.array(transformed["bboxes"])
            transformed_label = np.concatenate(to_merge, 1)

            images.append(transformed_image)
            labels_original.append(transformed_label)

        labels4 = []
        s = self.img_size // 2
        xc, yc = np.random.randint(s * 0.75, s * 1.25, (2,))  # mosaic center x, y
        indexes = np.random.permutation([0, 1, 2, 3])  # 3 additional image indices

        for placement_index, i in enumerate(indexes):
            img = images[i]
            h, w, _ = img.shape

            # place img in img4
            if placement_index == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif placement_index == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif placement_index == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif placement_index == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

            # Recalculate bounding boxes
            labels = labels_original[i].copy()
            if labels[0, 0] == -99:
                continue
            pad_w = x1a - x1b
            pad_h = y1a - y1b

            # Yolo to cartesian + padding
            x = labels_original[i]
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + pad_w  # top left x
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + pad_h  # top left y
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + pad_w  # bottom right x
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + pad_h  # bottom right y

            # Cartesian to yolo
            x = labels.copy()
            labels[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / img_size  # x center
            labels[:, 2] = ((x[:, 2] + x[:, 4]) / 2) / img_size  # y center
            labels[:, 3] = (x[:, 3] - x[:, 1]) / img_size  # width
            labels[:, 4] = (x[:, 4] - x[:, 2]) / img_size  # height

            labels4.append(labels)

        # Prepare the final mosaic image
        img4 = self.transform_post_mosaic(image=img4)["image"]

        # there's a non-zero chance that all 4 labels are empty. If so, will return a dummy bounding box
        if len(labels4) == 0:
            return img4, torch.tensor([[-99, 0.5, 0.5, 0.999, 0.999]])

        labels4 = np.concatenate(labels4)
        return img4, torch.tensor(labels4)


if __name__ == "__main__":
    # Setup testing
    image_path = "../dataset/data_final/data/egmont_11-12-2021_09.58_FILE0005_192.png"
    anno_path = "../dataset/data_final/data/egmont_11-12-2021_09.58_FILE0005_192.txt"
    image_bgr = cv2.imread(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    anno = U.experimental.read_yolo_annotation_file(anno_path).numpy()
    anno = np.array([[-1.    , 0.5, 0.5, 0.999, 0.999]])

    # Basic
    augment = BasicAugmentation(416)
    out = augment(image, anno)
    U.pytorch.show_tensor_image(out[0].unsqueeze(0))

    # Inference
    augment = InferenceAugmentation(416)
    out = augment(image, None)
    U.pytorch.show_tensor_image(out[0].unsqueeze(0))

    # Special
    augment = SpecialEffects(416, 416)
    out = augment(image, anno)
    U.pytorch.show_tensor_image(out[0].unsqueeze(0))

    # Special
    augment = MosaicAugmentation(416)
    out = augment([image]*4, [anno]*4)
    U.pytorch.show_tensor_image(out[0].unsqueeze(0))