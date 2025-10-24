import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset

stat = 1024
scale_values = [5, 5]
dtype = torch.float32


def load_cpcf_train():
    image_size = 128
    f = h5py.File(f"kolmogorov_train.h5", "r")

    all_fields = f["fields"][:]
    all_nominal = f["reynolds"][:]

    f.close()

    all_nominal = (all_nominal - 100) / 200 - 1
    all_nominal = all_nominal.repeat(stat, axis=0).reshape(-1, 1)

    for i in range(len(scale_values)):
        all_fields[:, :, i] = all_fields[:, :, i] / scale_values[i]
    all_fields = all_fields.reshape(-1, len(scale_values), image_size, image_size)

    tensor_nn_in = torch.tensor(all_nominal, dtype=dtype)
    tensor_nn_ou = torch.tensor(all_fields, dtype=dtype)

    nn_dataset = TensorDataset(tensor_nn_in, tensor_nn_ou)
    return nn_dataset


def load_cpcf_test():
    image_size = 128
    dataset_size = 16

    f = h5py.File(f"kolmogorov_test.h5", "r")

    all_fields = f["fields"][:]
    all_nominal = f["reynolds"][:]

    f.close()

    all_nominal = (all_nominal - 100) / 200 - 1
    all_nominal = all_nominal.reshape(-1, 1)

    for i in range(len(scale_values)):
        all_fields[:, :, i] = all_fields[:, :, i] / scale_values[i]

    nn_ou = np.zeros((dataset_size, 2, len(scale_values), image_size, image_size), dtype=np.float32)
    nn_ou[:, 0, :, :, :] = all_fields.mean(axis=1)
    nn_ou[:, 1, :, :, :] = all_fields.std(axis=1)

    tensor_nn_in = torch.tensor(all_nominal, dtype=dtype)
    tensor_nn_ou = torch.tensor(nn_ou, dtype=dtype)

    nn_dataset = TensorDataset(tensor_nn_in, tensor_nn_ou)
    return nn_dataset


if __name__ == "__main__":
    dataset = load_cpcf_train()
    dataset_test = load_cpcf_test()
