import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch.utils import data

p_mean = 0.0
p_std = 0.576
k_mean = 1.386
k_std = 10.64


def generalized_image_to_b_xy_c(tensor):
    """
    Transpose the tensor from [batch, channels, ..., pixel_x, pixel_y] to [batch, pixel_x*pixel_y, channels, ...]. We assume two pixel dimensions.
    """
    num_dims = len(tensor.shape) - 3  # Subtracting batch and pixel dimensions
    pattern = "b " + " ".join([f"c{i}" for i in range(num_dims)]) + " x y -> b (x y) " + " ".join([f"c{i}" for i in range(num_dims)])
    return rearrange(tensor, pattern)


def generalized_b_xy_c_to_image(tensor, pixels_x=None, pixels_y=None):
    """
    Transpose the tensor from [batch, pixel_x*pixel_y, channels, ...] to [batch, channels, ..., pixel_x, pixel_y] using einops.
    """
    if pixels_x is None or pixels_y is None:
        pixels_x = pixels_y = int(np.sqrt(tensor.shape[1]))
    num_dims = len(tensor.shape) - 2  # Subtracting batch and pixel dimensions (NOTE that we assume two pixel dimensions that are FLATTENED into one dimension)
    pattern = "b (x y) " + " ".join([f"c{i}" for i in range(num_dims)]) + f" -> b " + " ".join([f"c{i}" for i in range(num_dims)]) + " x y"
    return rearrange(tensor, pattern, x=pixels_x, y=pixels_y)


class DarcyDataset(data.Dataset):
    def __init__(
        self,
        data_directories,
        use_double=False,
        return_img=True,
        gaussian_prior=False,
    ):
        super().__init__()

        # Assuming data_directories is a tuple of file paths
        self.data_paths = list(data_directories)
        channels = len(self.data_paths)

        # load data
        for i in range(channels):
            if i == 0:
                self.data = pd.read_csv(self.data_paths[i], header=None)
            else:
                self.data = np.stack((self.data, pd.read_csv(self.data_paths[i], header=None)), axis=-1)

        self.data[:, :, 0] = (self.data[:, :, 0] - p_mean) / p_std
        self.data[:, :, 1] = (self.data[:, :, 1] - k_mean) / k_std

        # convert to torch tensor
        dtype = torch.float64 if use_double else torch.float32
        self.data = torch.tensor(self.data, dtype=dtype)
        self.num_datapoints = len(self.data)

        if return_img:
            assert len(self.data.shape) == 3, "Data must be of shape (num_datapoints, pixels_x*pixels_y, channels)"
            self.data = generalized_b_xy_c_to_image(self.data)

        if gaussian_prior:
            # instead consider no information at all
            self.data = torch.randn_like(self.data)

    def normalize(self, arr, min_val, max_val):
        return (arr - min_val) / (max_val - min_val)

    def unnorm(self, arr, min_val, max_val):
        return arr * (max_val - min_val) + min_val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if index >= self.num_datapoints:
            raise IndexError("index out of range")
        return self.data[index]


if __name__ == "__main__":
    darcy_train = DarcyDataset(("train/p_data.csv", "train/K_data.csv"))
    darcy_valid = DarcyDataset(("valid/p_data.csv", "valid/K_data.csv"))
