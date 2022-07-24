import random
import warnings

import albumentations
import cv2
import numpy as np
import natsort
import torchvision
from glob import glob
import os
import dutils as U
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from augmentations import BasicAugmentation, InferenceAugmentation, SpecialEffects, MosaicAugmentation
from bounding_boxes import read_yolo_from_file, yolo2cell
from general_utils import adjust_bb
import checker
import torch


class ShapeDataset:
    """ A simple toy dataset intended for debugging purposes """
    def __init__(self, S:int, img_size: int = None, inference:bool = None, mosaic:bool = None, special:bool = None, add_empty_frames:bool = False, include_robust:bool=True, basic:bool=True):
        """
        :param S: Height/width of the grid used to encode YOLO-annotations
        NOTE: img_size, inference, mosaic, special and add_empty_frames are just here for compatibility sake (TODO: make a shared parent class to inherited from instead)
        """

        # Checks
        checker.assert_types([S], [int])

        # Setup
        self.S = S
        self.C = 2
        self.class_int_to_name = {0:"T", 1:"C"}
        self.image_size = 256

        if inference:
            image_paths = natsort.natsorted(glob("../dataset/data_shape/train/images/*.png"))
            annotation_paths = natsort.natsorted(glob("../dataset/data_shape/train/labels/*.txt"))
        else:
            image_paths = natsort.natsorted(glob("../dataset/data_shape/valid/images/*.png"))
            annotation_paths = natsort.natsorted(glob("../dataset/data_shape/valid/labels/*.txt"))
        assert image_paths and annotation_paths and len(image_paths) == len(annotation_paths), "Shape mismatch"

        # Prepare images
        self.images = []
        for image_path in image_paths:
            image_bgr = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            final_image = torchvision.transforms.ToTensor()(image_rgb)
            self.images.append(final_image)

        # Prepare annotations
        self.label_matrices = []
        for annotations_path in annotation_paths:
            label_yolo = read_yolo_from_file(annotations_path)
            label_yolo = torch.tensor(label_yolo)
            label_matrix = yolo2cell(label_yolo, 256, S, self.C)
            self.label_matrices.append(label_matrix)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.label_matrices[i]


class MnistDataset:
    """ A simple toy dataset intended as a proof of concept """
    def __init__(self, S: int, img_size: int = None, inference:bool = None, mosaic:bool = None, special:bool = None, add_empty_frames:bool = False, include_robust:bool=True, basic:bool=True):
        """
        :param S: Height/width of the grid used to encode YOLO-annotations
        NOTE: img_size, inference, mosaic, special and add_empty_frames are just here for compatibility sake (TODO: make a shared parent class to inherited from instead)
        """

        # Checks
        checker.assert_types([S], [int])

        # Setup
        self.S = S
        self.C = 10
        self.class_int_to_name = {i:str(i) for i in range(10)}
        self.image_size = 256

        if inference:
            image_paths = natsort.natsorted(glob("../dataset/data_mnist/train/images/*.png"))
            annotation_paths = natsort.natsorted(glob("../dataset/data_mnist/train/labels/*.txt"))
        else:
            image_paths = natsort.natsorted(glob("../dataset/data_mnist/valid/images/*.png"))
            annotation_paths = natsort.natsorted(glob("../dataset/data_mnist/valid/labels/*.txt"))
        assert image_paths and annotation_paths and len(image_paths) == len(annotation_paths), "Shape mismatch"

        # Prepare images
        self.images = []
        for image_path in image_paths:
            image_bgr = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            final_image = torchvision.transforms.ToTensor()(image_rgb)
            self.images.append(final_image)

        # Prepare annotations
        self.label_matrices = []
        for annotation_path in annotation_paths:
            labels_yolo = []

            # Read file
            with open(annotation_path, "r") as f:
                raw = f.read()
            labels_corner = raw.strip().split("\n")[1:] # it's a weird format, not important

            # Convert labels from cartesian -> yolo -> cell
            for label in labels_corner:
                class_label, x1, y1, x2, y2 = map(int, label.split(","))

                w = (x2 - x1) / self.image_size
                h = (y2 - y1) / self.image_size
                x_center = x1 / self.image_size + w / 2
                y_center = y1 / self.image_size + h / 2
                labels_yolo.append([class_label, x_center, y_center, w, h])
            labels_yolo = torch.tensor(labels_yolo)
            label_matrix = yolo2cell(labels_yolo, self.image_size, S=S, C=self.C, B=1)
            self.label_matrices.append(label_matrix)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.label_matrices[i]


class CopenhagenDataset:
    """ A real, but small dataset intended as a last proof of concept """
    def __init__(self, S:int=12, img_size:int=396, inference:bool=False, mosaic:bool=False, special:bool=False, add_empty_frames:bool=False, include_robust:bool=True, basic:bool=True):
        """
        :param S: Height/width of the grid used to encode YOLO-annotations
        :param img_size: Size of images after resizing
        :param inference: If True will minimal preprocessing. If False will apply basic augmentation e.g. cutout
        :param mosaic: If True, will apply mosaic augmentation (ignored if `inference` is True)
        :param special: If True, will apply special effect augmentation (ignored if `inference` is True)

        NOTE: add_empty_frames is just here for the sake of compatibly
        """

        # Checks
        checker.assert_types([S, img_size, inference, add_empty_frames], [int, int, bool, bool])

        # Setup
        self.skipped_count = 0
        self.S = S
        self.C = 2
        self.B = 1
        self.img_size = img_size
        self.inference = inference
        self.add_empty_frames = add_empty_frames
        self.class_int_to_name = {0: "N", 1: "H"}

        # Config - inference or training
        if inference:
            self.inference_augmentation = InferenceAugmentation(img_size)
            image_paths = natsort.natsorted(glob("../dataset/data_copenhagen/train/images/*.png"))
            annotation_paths = natsort.natsorted(glob("../dataset/data_copenhagen/train/labels/*.txt"))
        else:
            self.basic_augmentation =  BasicAugmentation(img_size)
            self.mosaic_augmentation = MosaicAugmentation(img_size) if mosaic else None
            self.special_augmentation = SpecialEffects(img_size, img_size) if special else None
            image_paths = natsort.natsorted(glob("../dataset/data_copenhagen/valid/images/*.png"))
            annotation_paths = natsort.natsorted(glob("../dataset/data_copenhagen/valid/labels/*.txt"))

        # Check input files
        assert len(image_paths) and len(annotation_paths), "empty folder"
        assert len(image_paths) == len(annotation_paths), "Shape mismatch"

        # Prepare images
        self.images_numpy = []
        for image_path in image_paths:
            image_bgr = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self.images_numpy.append(image_rgb)

        # Prepare annotations
        self.labels_yolo = []
        for label_path in annotation_paths:
            label_yolo = read_yolo_from_file(label_path)
            label_yolo = [adjust_bb(l, self.img_size) for l in label_yolo] # move bb if it's close to the edge
            label_yolo = np.array(label_yolo)
            self.labels_yolo.append(label_yolo)

        assert len(self.labels_yolo) == len(self.images_numpy ), "Shape mismatch"


    def __len__(self):
        return len(self.images_numpy)


    def __getitem__(self, i):
        image_raw, label_raw = self.images_numpy[i], self.labels_yolo[i]

        if self.inference:
            image_torch_augmented = self.inference_augmentation(image_raw)
            label_yolo_augmented = torch.tensor(label_raw)
        elif self.mosaic_augmentation and (random.random() < 0.50):
            chosen_indexes = np.random.randint(0, self.__len__(), (3,)).tolist()
            images_4 = [image_raw] + [self.images_numpy[index] for index in chosen_indexes]
            labels_4 = [label_raw] + [self.labels_yolo[index]  for index in chosen_indexes]
            image_torch_augmented, label_yolo_augmented = self.mosaic_augmentation(images_4, labels_4, self.img_size)
        elif self.special_augmentation and (random.random() < 0.25):
            image_torch_augmented, label_yolo_augmented = self.special_augmentation(image_raw, label_raw)
        else:
            image_torch_augmented, label_yolo_augmented = self.basic_augmentation(image_raw, label_raw)

        # If the frame is empty (i.e. class_label == -99) the encoding will be done manually to avoid unnecessary complications
        if label_yolo_augmented[0, 0] == -99:
            label_cell_augmented = torch.zeros((self.S, self.S, 5 * self.B + self.C))
        else:
            label_cell_augmented = yolo2cell(label_yolo_augmented, self.img_size, self.S, self.C, self.B)

        return image_torch_augmented, label_cell_augmented


class ComparisonDataset:
    """ The final dataset """
    def __init__(self, S:int=12, img_size:int=348, inference:bool=False, mosaic:bool=False, special:bool=False, add_empty_frames:bool=False, robust:bool=True, basic:bool=True):
        """
        :param S: Height/width of the grid used to encode YOLO-annotations
        :param img_size: Size of images after resizing
        :param inference: If True will minimal preprocessing. If False will apply basic augmentation e.g. cutout
        :param mosaic: If True, will apply mosaic augmentation (ignored if `inference` is True)
        :param special: If True, will apply special effect augmentation (ignored if `inference` is True)
        :param add_empty_frames: If True, will add empty frames to the training images (ignored if `inference` is True)
        :param include_robust: If True, will add robustness frames to the training images.
        """

        # Checks
        checker.assert_types([S, img_size, inference, add_empty_frames], [int, int, bool, bool])
        if inference and any([basic, mosaic, special, add_empty_frames, robust]):
            warnings.warn("`inference == True` do not work together with any other augmentations. All of these (e.g. mosaic) will be ignored.")

        # Setup
        self.skipped_count = 0
        self.S = S
        self.C = 15
        self.B = 1
        self.img_size = img_size
        self.inference = inference
        self.add_empty_frames = add_empty_frames

        # Handle basic == False. NOTE: this is not something one would ever do in reality. But it's here for the sake of the report.
        self.basic_off = basic == False
        if self.basic_off:
            assert all([x==False for x in [mosaic, special, add_empty_frames, robust]]), "`basic` can only be turned off alone, the only reason it's possible to turn it off is for the report.`"

        # Config - inference or training
        if inference or self.basic_off:
            self.inference_augmentation = InferenceAugmentation(img_size)
        else:
            self.basic_augmentation =  BasicAugmentation(img_size) if basic else InferenceAugmentation(img_size)
            self.mosaic_augmentation = MosaicAugmentation(img_size) if mosaic else None
            self.special_augmentation = SpecialEffects(img_size, img_size) if special else None

        path_csv = f"../dataset/data_final/annotations_{'valid' if inference else 'train'}.csv"

        # Load annotations
        df_info = pd.read_csv(path_csv)

        # Mappings given by CVAT
        self.class_int_to_name = {
             0:'cycle_helmet',
             1:'cycle_nohelmet',
             2:'cycle_blurred',
             3:'cycle_covered',
             4:'escooter_helmet',
             5:'escooter_nohelmet',
             6:'escooter_blurred',
             7:'escooter_covered',
             8:'headphones',
             9:'earbuds',
             10:'phone',
             11:'hovding',
             12:'cycle_light',
             13:'escooter_light',
             14:'scooter'
        }
        # NOTE: This is sort af hack to deal with the last minut change to a 2 and 4 class setup
            # 2-class
        #self.class_int_to_name = {0:"cycle", 1:"escooter"}
        #path_csv = f"../dataset/data_final/annotations_{'valid' if inference else 'train'}_2class_clean.csv"
            # 4-class
        self.class_int_to_name = {0: "Cyclist helmet", 1: "Cyclist no helmet", 2:"Escooter helmet", 3:"Escooter no helmet"}
        mapper = {"0": "0", "1": "1", "4": "2", "5": "3"}
        df_info["label_yolo_combined"] = df_info["label_yolo_combined"].apply(lambda x: " ".join([mapper[x.split()[0]]] + x.split()[1:]) ) # remap the YOLO labels to be a 4-class setup


        # Include/remove robustness dataset
        if not robust:
            df_info = df_info[df_info["location"] != "internet"]

        # Read images
        image_paths = df_info["frame_path"].unique().tolist()
        assert len(image_paths) == len(df_info["annotation_name"].unique()), "Shape mismatch"
        for path in image_paths:
            if not os.path.exists(path):
                print(path)

        # Prepare images
        self.images_numpy = []
        for image_path in image_paths:
            image_bgr = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self.images_numpy.append(image_rgb)

        # Prepare annotations
        self.labels_yolo = []
        for annotation_name in list(df_info["annotation_name"].unique()):
            label_yolo = df_info.loc[df_info["annotation_name"] == annotation_name, "label_yolo_combined"].tolist()
            label_yolo = [list(map(float, l.split())) for l in label_yolo] # from yolo string to list
            label_yolo = [adjust_bb(l, self.img_size) for l in label_yolo] # move bb if it's close to the edge
            label_yolo = np.array(label_yolo)
            self.labels_yolo.append(label_yolo)
        assert min([label[:,1:].min() for label in self.labels_yolo]) >= 0.0, "One or more YOLO annotations have values below 0"
        assert min([label[:,1:].max() for label in self.labels_yolo]) >= 0.0, "One or more YOLO annotations have values above 1"


        # Add empty frames.
        if (not self.inference) and self.add_empty_frames:
            # images
            self.empty_frame_paths = glob("../dataset/data_final/empty_frames_downscaled/*")
            self.images_numpy += [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in self.empty_frame_paths]

            # Annotations
            #     Dummy annotations given to empty frame. This is not intuitive, but -99 i used to encode empty_frame
            #     and bbox values are chosen to be the full image to avoid `RandomSizedBBoxSafeCrop` taken away to much
            self.empty_frame_annotations = [np.array([[-99, 0.5, 0.5, 0.999, 0.999]])] * len(self.empty_frame_paths)
            self.labels_yolo += self.empty_frame_annotations

        # Wrap up
        assert len(self.labels_yolo) == len(self.images_numpy), "Shape mismatch"
        self.df_info = df_info


    def __len__(self):
        return len(self.images_numpy)


    def __getitem__(self, i):
        image_raw, label_raw = self.images_numpy[i], self.labels_yolo[i]

        if self.basic_off or self.inference: # If inference or basic augmentation is of
            image_torch_augmented = self.inference_augmentation(image_raw)
            label_yolo_augmented = torch.tensor(label_raw)
        elif self.mosaic_augmentation and (random.random() < 0.50):
            chosen_indexes = np.random.randint(0, self.__len__(), (3,)).tolist()
            images_4 = [image_raw] + [self.images_numpy[index] for index in chosen_indexes]
            labels_4 = [label_raw] + [self.labels_yolo[index]  for index in chosen_indexes]
            image_torch_augmented, label_yolo_augmented = self.mosaic_augmentation(images_4, labels_4, self.img_size)
        elif self.special_augmentation and (random.random() < 0.25):
            image_torch_augmented, label_yolo_augmented = self.special_augmentation(image_raw, label_raw)
        else:
            image_torch_augmented, label_yolo_augmented = self.basic_augmentation(image_raw, label_raw)

        # If the frame is empty (i.e. class_label == -99) the encoding will be done manually to avoid unnecessary complications
        if label_yolo_augmented[0, 0] == -99:
            label_cell_augmented = torch.zeros((self.S, self.S, 5 * self.B + self.C))
        else:
            label_cell_augmented = yolo2cell(label_yolo_augmented, self.img_size, self.S, self.C, self.B)

        return image_torch_augmented, label_cell_augmented


if __name__ == "__main__":
    dataset = ComparisonDataset(12, img_size=240, inference=False, mosaic=False, special=False, add_empty_frames=False, robust=False, basic=False)
    dataset = ComparisonDataset(12, img_size=240, inference=True, mosaic=False, special=False, add_empty_frames=False, robust=False, basic=False)
    1/0
    # Set "train" --> "valid" when you run this. It takes an entirety otherwise.
    for m in [True, False]:
        for s in [True, False]:
            for a in [True, False]:
                for r in [True, False]:
                    print(m,s,a,r)
                    dataset = ComparisonDataset(12, img_size=240, inference=False, mosaic=m, special=s, add_empty_frames=a, robust=r, basic=True)
                    for i in range(len(dataset)):
                        image, anno = dataset[i]

