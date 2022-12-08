import os

import numpy as np
import torch

from data_utils import max_size, normalize, purify
from hparms import HParams
from model import Model


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hp = HParams()

    dataset = np.load(hp.data_location, encoding="latin1", allow_pickle=True)
    data = dataset["train"]
    data = purify(data, hp)
    data = normalize(data)
    N_max = max_size(data)

    output_folder = "output_2"
    ckpt_save_path = os.path.join(output_folder, "ckp")
    image_save_path = os.path.join(output_folder, "img")
    os.makedirs(ckpt_save_path, exist_ok=True)
    os.makedirs(image_save_path, exist_ok=True)
    model = Model(ckpt_save_path, image_save_path, hp, N_max, device)
    for iter in range(50001):
        model.train(iter, data)

    # """
    # sample_iter = 50000
    # model.load(f"output/ckp/encoder_iter_{sample_iter:06d}.pth", f"output/ckp/decoder_iter_{sample_iter:06d}.pth")
    # model.conditional_generation(data, 50000)
    # """
