import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch


def max_size(data):
    """larger sequence length in the data set"""
    sizes = [len(seq) for seq in data]
    return max(sizes)


def purify(strokes, hp):
    """removes to small or too long sequences + removes large gaps"""
    data = []
    for seq in strokes:
        if seq.shape[0] <= hp.max_seq_length and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data


def calculate_normalizing_scale_factor(strokes):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(strokes)):
        for j in range(len(strokes[i])):
            data.append(strokes[i][j, 0])
            data.append(strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)


def normalize(strokes):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    data = []
    scale_factor = calculate_normalizing_scale_factor(strokes)
    for seq in strokes:
        seq[:, 0:2] /= scale_factor
        data.append(seq)
    return data


def make_batch(data, batch_size, N_max, device):
    batch_idx = np.random.choice(len(data), batch_size)
    batch_sequences = [data[idx] for idx in batch_idx]
    strokes = []
    lengths = []
    index = 0
    for seq in batch_sequences:
        len_seq = len(seq[:, 0])
        new_seq = np.zeros((N_max, 5))
        new_seq[:len_seq, :2] = seq[:, :2]
        new_seq[: (len_seq - 1), 2] = 1 - seq[:-1, 2]
        new_seq[:len_seq, 3] = seq[:, 2]
        new_seq[(len_seq - 1) :, 4] = 1
        new_seq[len_seq - 1, 2:4] = 0
        lengths.append(len(seq[:, 0]))
        strokes.append(new_seq)
        index += 1

    batch = torch.from_numpy(np.stack(strokes, 1)).to(device).float()
    return batch, lengths


def sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, hp, greedy=False):
    # inputs must be floats
    if greedy:
        return mu_x, mu_y
    mean = [mu_x, mu_y]
    sigma_x *= np.sqrt(hp.temperature)
    sigma_y *= np.sqrt(hp.temperature)
    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y], [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def make_image(sequence, iter, img_output_path):
    """plot drawing with separated strokes"""
    strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0] + 1)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    img_save_path = os.path.join(img_output_path, f"{iter:06d}.png")
    pil_image.save(img_save_path, "JPEG")
    plt.close("all")
