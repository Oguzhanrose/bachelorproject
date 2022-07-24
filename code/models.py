import timm
import torch
from checker import assert_types
from torch import nn
from general_utils import non_max_suppression
from metrics import mean_batch_map
from plots import draw_bb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os.path
import numpy as np
import torch
import cv2
from dutils.jupyter_ipython import show_image as show

########################################################################################################################
#                                                       DARKNET 53
########################################################################################################################


def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, block, num_classes):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


########################################################################################################################
#                                                       YOLOv1
########################################################################################################################


class YoloModel(nn.Module):
    """ A modified version of YOLOv1"""

    def __init__(self, S:int=2, C:int=2, B:int=1, backbone:str="efficient_net", image_size:int=348) -> None:
        """
        :param S: Height/width of the grid used to encode YOLO-annotations
        :param C: Number of classes used for one hot encoding
        :param B: Number of bounding boxes predicted by the model
        """
        # Checks
        assert_types([S, C, B, backbone, image_size], [int, int, int, str, int])
        assert backbone in ["darknet53", "efficient_net"], "`backbone` most be `darknet53` or `efficient_net`"

        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.outdim = 5 * self.B + C
        self.image_size = image_size

        self.color_map = {
            0: (31, 119, 180),
            1: (255, 127, 14),
            2: (44, 160, 44),
            3: (214, 39, 40),
            4: (148, 103, 189),
            5: (140, 86, 75),
            6: (227, 119, 194),
            7: (127, 127, 127),
            8: (188, 189, 34),
            9: (23, 190, 207),
            10: [145, 68, 148],
            11: [124, 103, 65],
            12: [89, 64, 138],
            13: [124, 62, 84],
            14: [105, 111, 120]
        }

        if backbone == "efficient_net":
            self.model = timm.create_model("efficientnet_b0", pretrained=True)
            self.model.classifier = torch.nn.Linear(1280, S * S * self.outdim)

        elif backbone == "darknet53":
            self.model = Darknet53(DarkResidualBlock, 1000)
            self.model.load_state_dict(torch.load("../saved_models/darknet53.pth"))
            self.model.fc = torch.nn.Linear(1024, S * S * self.outdim)


    def forward(self, images) -> torch.Tensor:
        """
        :param images: torch squared image of shape (batch_size, color_channels, height, width)
        :return: torch tensor of shape (batch_size, S, S, 5*B+C)
        """
        out = self.model(images)
        out = out.view(-1, self.S, self.S, self.outdim)
        out[..., 0:3] = out[..., 0:3].sigmoid()
        return out


    def inference(self, images, config):
        """
        :param images:
        :param config:
        :return:
        """

        # Checks
        assert_types([images], [torch.Tensor])
        assert len(images.shape) == 4 and images.shape[1] == 3, "Received bad image batch. Did you forget to pass a batch?"
        assert images.shape[2] == images.shape[3] == self.image_size, "Wrong image size. Did you forget augmentation?"

        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            outs_cell = self.forward(images)
        outs_cell[..., 3:5] = torch.abs(outs_cell[..., 3:5]) # We don't have the square part as in the loss function, so I'm gonna resolve the problem of negative widths and heights with abs()

        # Non-max suppression
        to_return = []
        for out_cell in outs_cell:
            nms_bbs_corner = non_max_suppression(out_cell, self.image_size, self.C, config.nms_conf_threshold, config.nms_iou_threshold, config.class_sensitive)
            to_return.append(nms_bbs_corner)

        return to_return

    def _draw_images(self, images, config):
        preds_batch = self.inference(images, config)

        # plot image
        final_images = []
        for (image, preds) in zip(images, preds_batch):
            image = image.clone()
            image = (image.permute(1, 2, 0).detach().cpu() * 255).numpy().astype(np.uint8)
            for (conf, x1, y1, x2, y2, class_label) in preds:
                bb = [int(c) for c in [x1, y1, x2, y2]]
                class_label = int(class_label)
                image = draw_bb(image, bb, config.class_int_to_name[class_label], conf, self.color_map[class_label])

            final_images.append(image)
        return final_images


    def plot_images(self, images, config):
        final_images = self._draw_images(images, config)
        # `show` will be imported in the notebook, it's a jupyter notebook only library and as such cannot be imported here
        return show(final_images, BGR2RGB=False, return_image=True)


    def video_inference(self, video_path:str, video_save_path:str, config) -> None:
        # Checks
        assert os.path.exists(video_path), "Bad video path"

        # Augmentation
        augmentation = A.Compose([
            A.Resize(config.image_size, config.image_size),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2()
        ])

        # Break video into frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            video_feed_active, frame = cap.read()

            if not video_feed_active:
                cap.release()
                break

            frames.append(frame)

        # devide frames into batches
        frames_to_batch_all = []
        for i in range(int(np.ceil(len(frames) / config.batch_size))):
            frames_to_batch = []
            for frame in frames[(i * config.batch_size):(i + 1) * config.batch_size]:
                frame_augmented = augmentation(image=frame)["image"]
                frames_to_batch.append(frame_augmented)

            corrected = torch.stack(frames_to_batch).to(config.device)
            frames_to_batch_all.append(corrected)

        # Inference
        final_images = []
        for images in frames_to_batch_all:
            images_drawn = self._draw_images(images, config)
            for image in images_drawn:
                final_images.append(image)

        # Make into video
        image = final_images[0]
        h, w, c = image.shape
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(video_save_path, fourcc, 5, (w, h))
        [video.write(image) for image in final_images]
        video.release()

        return None




_ce = nn.CrossEntropyLoss()
def yolo_loss(out:torch.Tensor, targets:torch.Tensor, config, calc_map:bool) -> torch.Tensor:
    # setup
    loss_xy = torch.tensor(0.).to(config.device)
    loss_wh = torch.tensor(0.).to(config.device)
    loss_object = torch.tensor(0.).to(config.device)
    loss_no_object = torch.tensor(0.).to(config.device)
    loss_class = torch.tensor(0.).to(config.device)

    S, lambda_coord, lambda_noobj = config.S, config.lambda_coord, config.lambda_noobj

    if targets is None:
        targets = torch.zeros(out.shape)

    for b in range(out.shape[0]):
        for i in range(S):
            for j in range(S):

                # no_obj
                if targets[b, i, j, 0] == 0:
                    loss_no_object += out[b, i, j, 0] ** 2
                    continue

                # xy
                x_hat = out[b, i, j, 1]
                y_hat = out[b, i, j, 2]
                x = targets[b, i, j, 1]
                y = targets[b, i, j, 2]
                loss_xy += (x_hat - x) ** 2 + (y_hat - y) ** 2

                # wh
                w_hat = torch.sqrt(torch.abs(out[b, i, j, 3]))
                h_hat = torch.sqrt(torch.abs(out[b, i, j, 4]))
                w = torch.sqrt(targets[b, i, j, 3])
                h = torch.sqrt(targets[b, i, j, 4])
                loss_wh += (h_hat - h) ** 2 + (w_hat - w) ** 2

                # obj
                loss_object += (1 - out[b, i, j, 0]) ** 2

                # class
                loss_class += _ce(out[b, i, j, 5:].unsqueeze(0), targets[b, i, j, 5:].argmax().long().unsqueeze(0))

    loss = lambda_coord * loss_xy + \
           lambda_coord * loss_wh + \
           loss_object + \
           lambda_noobj * loss_no_object + \
           loss_class
    
    losses = {
        "combined": loss,
        "xy": loss_xy * lambda_coord,
        "wh": loss_wh * lambda_coord,
        "object": loss_object,
        "no_object": lambda_noobj * loss_no_object,
        "class": loss_class,
        "map_average": torch.tensor(0),
        "map_50": torch.tensor(0),
        "map_75": torch.tensor(0),
        "map_95": torch.tensor(0),
    }

    # mAP
    if calc_map:
        with torch.no_grad():
            map_scores = mean_batch_map(out, targets, config)
            losses["map_average"] = torch.tensor(map_scores["average_over_all_thresholds"])
            losses["map_50"] = torch.tensor(map_scores[0.5])
            losses["map_75"] = torch.tensor(map_scores[0.75])
            losses["map_95"] = torch.tensor(map_scores[0.95])

    return losses
