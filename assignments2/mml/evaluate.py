"""
    Script to evaluate the model on the whole test set and save the results in folder.
"""

import argparse
import os
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import random_split
from tqdm import tqdm

from mml.data import ImageCaptionDataset
from mml.model import Net
from mml.utils import ConfigS, ConfigL
# from mml.utils import ConfigS, ConfigL, download_weights  # TODO:原始版本，可能download_weights有错或者需要自己实现

from matplotlib import rcParams
# 设置字体为支持全角符号的字体，如SimHei或其他常见中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 你可以根据需要选择合适的字体
rcParams['axes.unicode_minus'] = False  # 防止负号显示不正常

parser = argparse.ArgumentParser()

parser.add_argument(
    "-C", "--checkpoint-name", type=str, default="model.pt", help="Checkpoint name"
)

parser.add_argument(
    "-S",
    "--size",
    type=str,
    default="S",
    help="Model size [S, L]",
    choices=["S", "L", "s", "l"],
)

parser.add_argument(
    "-I", "--img-path", type=str, default="", help="Path to the test image folder"
)

parser.add_argument(
    "-R", "--res-path", type=str, default="", help="Path to the results folder"
)

parser.add_argument(
    "-T", "--temperature", type=float, default=1.0, help="Temperature for sampling"
)

args = parser.parse_args()

config = ConfigL() if args.size.upper() == "L" else ConfigS()

ckp_path = os.path.join(config.weights_dir, args.checkpoint_name)

assert os.path.exists(args.img_path), "Path to the test image folder does not exist"

# set seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True

is_cuda = torch.cuda.is_available()
device = "cuda" if is_cuda else "cpu"


def evaluate_dataset(model, dataset, img_path, save_path, temperature=1.0):
    """
    Evaluate model on dataset.

    Args:
        model: model to evaluate
        dataset: dataset to evaluate on
        img_path: path to images
        save_path: path to save results
    """
    model.eval()

    loop = tqdm(dataset, total=len(dataset))
    for img_name, _, _ in loop:
        img = Image.open(os.path.join(img_path, img_name))

        with torch.no_grad():
            caption, _ = model(img, temperature)

        plt.imshow(img)
        plt.title(caption)
        plt.axis("off")

        plt.savefig(os.path.join(save_path, img_name), bbox_inches="tight")

        plt.clf()
        plt.close()


if __name__ == "__main__":
    model = Net(
        clip_model=config.clip_model,
        text_model=config.text_model,
        ep_len=config.ep_len,
        num_layers=config.num_layers,
        n_heads=config.n_heads,
        forward_expansion=config.forward_expansion,
        dropout=config.dropout,
        max_len=config.max_len,
        device=device,
    )
    # TODO: 需要你自己实现一个ImageCaptionDataset在`data/dataset.py`中
    dataset = ImageCaptionDataset(
            image_dir="./datasets/train2014",
            caption_pth="./data/coco/annotations/train_caption.json",
            img_already_embedded=True
        )

    config.train_size = int(config.train_size * len(dataset))
    config.val_size = int(config.val_size * len(dataset))
    config.test_size = len(dataset) - config.train_size - config.val_size

    _, _, test_dataset = random_split(
        dataset, [config.train_size, config.val_size, config.test_size]
    )

    if not os.path.exists(config.weights_dir):
        os.makedirs(config.weights_dir)

    # if not os.path.isfile(ckp_path):
        # download_weights(ckp_path, args.size)

    checkpoint = torch.load(ckp_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])   

    save_path = os.path.join(
        args.res_path, f"{args.checkpoint_name[:-3]}_{args.size.upper()}"
    )

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    evaluate_dataset(model, test_dataset, args.img_path, save_path, args.temperature)
