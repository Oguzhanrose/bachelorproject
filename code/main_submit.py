
#####################################################################################################################
# 													Imports
#####################################################################################################################

import dutils as U
from dutils.jupyter_ipython import show_image as show

# Ours - in project
from datasets import ShapeDataset, MnistDataset, CopenhagenDataset, ComparisonDataset
from models import yolo_loss, YoloModel
from general_utils import seed_torch

# Normal imports
from tqdm import tqdm
import torch
import inspect
import numpy as np
import os, sys
import cv2
import pandas as pd
import seaborn as sns
import argparse
import shutil
import wandb

# Settings
# torch.set_printoptions(sci_mode=False)
# sns.set_style("whitegrid");


#####################################################################################################################
# 														Setup - Arguments
#####################################################################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="Testing", help='Name of the project')

parser.add_argument('--basic', type=int, default=1, help='If True, apply basic augmentation. If False, apply only inference augmentation.')
parser.add_argument('--special', type=int, default=1, help='If True, apply special augmentation.')
parser.add_argument('--mosaic', type=int, default=1, help='If True, apply mosaic augmentation.')
parser.add_argument('--empty_frames', type=int, default=1, help='Adds empty frames to the training data.')
parser.add_argument('--robust', type=int, default=1, help='Adds robust frames to the training data')
parser.add_argument('--img_size', type=int, default=348, help='The image size apply by the augmentation')
parser.add_argument('--batch_size', type=int, default=40, help='Batch size used in the training dataloader')
parser.add_argument('--start_lr', type=float, default=1e-3, help='Learning used in scheduler: start_lr * 0.99^(epoch)')

parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
parser.add_argument('--S', type=int, default=12, help='Number of grid splits')
args = parser.parse_args()


#####################################################################################################################
# 														Setup - Config
#####################################################################################################################

RUN_PATH = f"../runs/{args.name}"

class Config(torch.nn.Module): # torch.nn.Module is just a because it's easier to save the config file this wy
    # Control
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_model = True
    save_only_best_model = True
    plot_example_images_every_nth_epoch = 5

    # Dataset
    dataset = ComparisonDataset
    class_int_to_name = None
    image_size = args.img_size
    basic_augmentation = args.basic == 1
    special_augmentation = args.special == 1
    mosaic_augmentation = args.mosaic == 1
    add_empty_frames = args.empty_frames == 1
    add_robust_frames = args.robust == 1


    # Hypers - YOLO
    lambda_noobj = 0.5
    lambda_coord = 5.0
    S = args.S
    B = 1
    C = None

    # Hypers - mAP and NMS
    nms_conf_threshold = 0.35
    nms_iou_threshold = 0.80
    class_sensitive = False
    map_iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # --> mAP@0.5:0.05:0.95
    apply_nms_before_map = True

    # Hypers - General
    backbone = "darknet53"  # "efficient_net"
    batch_size = args.batch_size
    epochs = args.epochs
    epochs_trained = 0  # TODO: automate this i.e. read epochs_trained from model_load_path
    criterion = yolo_loss
    optimizer_hyper = dict(lr=args.start_lr)
    optimizer = torch.optim.Adam
    scheduler_hyper = dict(lr_lambda=lambda epoch: 0.99 ** epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR

    # Seed everything
    seed = 12
    seed_torch(seed=seed, deterministic=False)

config = Config()
assert not os.path.exists(RUN_PATH), "The name of the run has already been used. Rename `config.run_name` to something else"
assert all([t in config.map_iou_thresholds for t in [0.5, 0.75, 0.95]]), "These 3 values most be present because other parts of the code assume they are"
assert config.image_size % config.S == 0, "`image_size` must be divisible by `S`"


#####################################################################################################################
# 									           Setup - prepare run folder
#####################################################################################################################


# Setup local logging folder
os.mkdir(RUN_PATH)
os.mkdir(os.path.join(RUN_PATH, "images"))

config_class_source_code = inspect.getsource(Config)
arguments = str(args)
with open(os.path.join(RUN_PATH, "config.txt"), 'x') as file:
    print(config_class_source_code + "\n" + arguments, file=file, end="")


#####################################################################################################################
# 												Dataset and DataLoader
#####################################################################################################################

# Dataset
train_dataset = config.dataset(
    config.S,
    img_size=config.image_size,
    mosaic=config.mosaic_augmentation,
    special=config.special_augmentation,
    add_empty_frames=config.add_empty_frames,
    robust = config.add_robust_frames,
    basic = config.basic_augmentation
)
valid_dataset = config.dataset(config.S, img_size=config.image_size, inference=True)

# Sanity check
i, a = train_dataset[0]
config.C = train_dataset.C
config.class_int_to_name = train_dataset.class_int_to_name
print(i.shape, a.shape)
del i, a


# DataLoader
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=13, shuffle=True) # shuffle for plotting. 13 because there's 91 valididation images

# Sanity check
bi, ba = next(iter(train_dl))
print(bi.shape, ba.shape)
show([image for image in bi[:8, ...]])

#####################################################################################################################
# 											  Models, optimizer and schedular
#####################################################################################################################


model = YoloModel(S=config.S, C=config.C, B=config.B, backbone=config.backbone, image_size=config.image_size)
model = model.to(config.device)
optimizer = config.optimizer(model.parameters(), **config.optimizer_hyper)
scheduler = config.scheduler(optimizer, **config.scheduler_hyper)

columns = ["lr"]
for name in ["combined", "xy", "wh", "object", "no_object", "class", "map_average", "map_50", "map_75", "map_95"]:
    columns.append("train_" + name)
    columns.append("valid_" + name)

stats = pd.DataFrame(np.zeros((config.epochs, len(columns))), columns=columns)
best_model_name = "0_EPOCH.pth"

# Final Sanity check before training
with torch.no_grad():
    out = model(bi.cuda())
print(out.shape)
del bi, ba


#####################################################################################################################
# 											  			Training
#####################################################################################################################

# Change working directory and spin up wandb
os.chdir(RUN_PATH)
wandb.login(key="5f35577dd337c909579d63ec11b3184d93aeedd7")
wandb.init(project="final_new", name=args.name, config={"info_combined":config_class_source_code + "\n" + arguments}, entity="bachelor")


for epoch in tqdm(range(config.epochs_trained, config.epochs+1)):
    train_losses = {"combined": 0, "xy": 0, "wh": 0, "object": 0, "no_object": 0, "class": 0, "map_average": 0,
                    "map_50": 0, "map_75": 0, "map_95": 0}
    valid_losses = {"combined": 0, "xy": 0, "wh": 0, "object": 0, "no_object": 0, "class": 0, "map_average": 0,
                    "map_50": 0, "map_75": 0, "map_95": 0}
    wandb_log = {"lr": optimizer.param_groups[0]["lr"]}

    model.train()
    for i, (images, labels) in enumerate(tqdm(train_dl, leave=True)):
        images = images.to(config.device)
        if labels is not None:
            labels = labels.to(config.device)

        # Forward pass
        preds = model(images)
        loss_info = yolo_loss(preds, labels, config, calc_map=epoch in [10 * i for i in range(1, 101)])
        loss = loss_info["combined"]

        # Backward pass
        loss.backward()

        # Batch update and logging
        optimizer.step()
        optimizer.zero_grad()

        for key in train_losses.keys():
            train_losses[key] += loss_info[key].detach().cpu().item() / len(train_dl)

    # Plotting some examples
    if epoch % config.plot_example_images_every_nth_epoch == 0:
        samples = min(len(images), 4)
        images_drawn = model.plot_images(images[:samples], config)
        cv2.imwrite("./images/train_image_" + str(epoch) + ".jpg", cv2.cvtColor(images_drawn, cv2.COLOR_BGR2RGB))
        wandb_log["train_examples"] = wandb.Image(images_drawn, caption=f"Train images")


    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(valid_dl, leave=True)):
            images = images.to(config.device)
            if labels is not None:
                labels = labels.to(config.device)

            # Forward pass
            preds = model(images)
            loss_info = yolo_loss(preds, labels, config, calc_map=epoch > 10)

            # Batch update and logging
            for key in valid_losses.keys():
                valid_losses[key] += loss_info[key].detach().cpu().item() / len(valid_dl)

    if optimizer.param_groups[0]["lr"] > 5e-5:
        scheduler.step()
    config.epochs_trained += 1

    # Plotting some examples
    if epoch % config.plot_example_images_every_nth_epoch == 0:
        samples = min(len(images), 4)
        images_drawn = model.plot_images(images[:samples], config)
        cv2.imwrite(f"./images/valid_image_" + str(epoch) + ".jpg", cv2.cvtColor(images_drawn, cv2.COLOR_BGR2RGB))
        wandb_log["valid_examples"] = wandb.Image(images_drawn, caption=f"Valid images")

    # Epoch update and logging
    for keys, value in train_losses.items():  # Adds training stats
        wandb_log["train_" + keys] = value
    for keys, value in valid_losses.items():  # Adds valid stats
        wandb_log["valid_" + keys] = value
    for key, value in wandb_log.items():
        if key in ["train_examples", "valid_examples"]: continue
        stats.loc[epoch, key] = value
    wandb.log(wandb_log)

    if (epoch > 0) and (stats["valid_map_average"][epoch] > stats.loc[:(epoch - 1),
                                                         "valid_map_average"].min()):  # Save model if it's better
        if config.save_model and config.save_only_best_model and (best_model_name != "0_EPOCH.pth"):
            U.input_output.remove_file(best_model_name)

        extra_info = {"valid_map_average": stats["valid_map_average"][epoch], "epochs_trained": config.epochs_trained}
        best_model_name = U.pytorch.get_model_save_name("model.pth", extra_info, include_time=True)
        if config.save_model: torch.save(model.state_dict(), best_model_name)

    stats.to_csv("stats.csv", index=False)
    torch.save({'model_state_dict': model.state_dict()}, "latest.pth")


# Save the best and the latest model + some extra data
shutil.copy(best_model_name, os.path.join(wandb.run.dir, best_model_name))
shutil.copy("latest.pth", os.path.join(wandb.run.dir, "latest.pth"))
wandb.save(best_model_name, policy="now")


#####################################################################################################################
# 											  	Inference - Training
#####################################################################################################################


for i, (images, labels) in enumerate(train_dl):
    images, labels = images.to(config.device), labels.to(config.device)
    images_drawn = model.plot_images(images, config)
    cv2.imwrite(f"./images/train_eval_final_{i}.png", cv2.cvtColor(images_drawn, cv2.COLOR_BGR2RGB))
    shutil.copy(f"./images/train_eval_final_{i}.png", os.path.join(wandb.run.dir, f"train_eval_final_{i}.png"))
    if i == 3: break


#####################################################################################################################
# 											  	Inference - validation
#####################################################################################################################

# Use best model
# model.load_state_dict(torch.load(best_model_name))
# model.eval()



for i, (images, labels) in enumerate(valid_dl):
    images, labels = images.to(config.device), labels.to(config.device)
    images_drawn = model.plot_images(images, config)
    cv2.imwrite(f"./images/valid_eval_final_{i}.png", cv2.cvtColor(images_drawn, cv2.COLOR_BGR2RGB))
    shutil.copy(f"./images/valid_eval_final_{i}.png", os.path.join(wandb.run.dir, f"valid_eval_final_{i}.png"))
    if i == 3: break


for in_path in ["../../video_data/valid_video.mp4", "../../video_data/test_video.mp4", "../../video_data/lyngbyvej_07-12-2021_14.13_FILE0056_233.MP4", "../../video_data/valby_14-02-2022_13.41_FILE0050.MP4"]:
    save_name = os.path.basename(in_path)
    model.video_inference(in_path, save_name, config)
    shutil.copy(save_name, os.path.join(wandb.run.dir, save_name))


# Closing wandb and removing its folder
wandb.finish()
shutil.rmtree("./wandb")























