import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="Testing", help='Name of the project')
parser.add_argument('--gpu_cluster', type=str, default="gpuv100", help='DTU GPU cluster https://www.hpc.dtu.dk/?page_id=2759')
parser.add_argument('--mail', type=str, default="s173174@student.dtu.dk", help='HPC mails info to this email')

parser.add_argument('--basic', type=int, default=1, help='If True, apply basic augmentation. If False, apply only inference augmentation.')
parser.add_argument('--special', type=int, default=1, help='If True, apply special augmentation.')
parser.add_argument('--mosaic', type=int, default=1, help='If True, apply mosaic augmentation.')
parser.add_argument('--empty_frames', type=int, default=1, help='Adds empty frames to the training data.')
parser.add_argument('--robust', type=int, default=1, help='Adds robust frames to the training data')
parser.add_argument('--img_size', type=int, default=348, help='The image size apply by the augmentation')
parser.add_argument('--batch_size', type=int, default=40, help='Batch size used in the training dataloader')
parser.add_argument('--start_lr', type=float, default=1e-3, help='Learning used in scheduler: start_lr * 0.99^(epoch)')

parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--S', type=int, default=12, help='Number of grid splits')
args = parser.parse_args()


bash_string = f"""\
#!/bin/sh
#BSUB -q {args.gpu_cluster}
#BSUB -J {args.name}
#BSUB -W 24:00
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB]"
#BSUB {args.mail}
#BSUB -B
#BSUB -N
#BSUB -o ../runs/{args.name}/output_%J.out
#BSUB -e ../runs/{args.name}/error_%J.out

nvidia-smi
python3 main_submit.py --name {args.name} --basic {args.basic} --special {args.special} --mosaic {args.mosaic} --empty_frames {args.empty_frames} --robust {args.robust} --img_size {args.img_size} --batch_size {args.batch_size} --start_lr {args.start_lr} --epochs {args.epochs} --S {args.S}
"""
submit_name = f"submit_{args.name}.sh"
with open(submit_name, "x") as file:
	print(bash_string, file=file, end="")

command = f"bsub < {submit_name}"
os.system(command)
#os.remove(submit_name)


#--name "nothing_on" --basic 1 --special 1 --mosaic 1 --empty_frames 1 --robust 1
#--name "nothing_on" --basic 0 --special 0 --mosaic 0 --empty_frames 0 --robust 0

