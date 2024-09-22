import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

matplotlib.use("Agg")


def plot_spectrogram(spectrogram, save_to: str = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spectrogram.T, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    if save_to:
        plt.savefig(save_to)
    plt.close()
    return fig


def save_array(spectrogram: torch.Tensor, path="./track_tensor.txt", fmt="%.4f"):
    """save 1-D or 2-D array to txt."""
    np_data = spectrogram.cpu().detach().numpy()
    np.savetxt(path, np_data, fmt=fmt)
