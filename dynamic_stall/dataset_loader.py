import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset

stat = 32
mean_values = [100000, 10000000, 400000, 290, 1.17, -2.55]
std_values = [80000, 18000000, 20000000, 56, 0.8, 400]
min_values = [5, 3, 10, 5]
max_values = [10, 5, 20, 10]
R = 8.31446261815324 / 28.966 * 1000
dtype = torch.float32


def load_cpcf_train(full_conditioning: bool, dataset_size: int):
    image_size = 128
    dataset_size = 128

    f = h5py.File(f"dynamic_stall_train.h5", "r")

    all_fields = f["fields"][:].reshape(dataset_size * stat, 8, image_size, image_size)
    all_nominal = f["nominal_condition"][:].reshape(dataset_size * stat, 4)

    f.close()

    for i in range(len(min_values)):
        all_nominal[:, i] = (all_nominal[:, i] - min_values[i]) / (max_values[i] - min_values[i])

    for i in range(len(mean_values)):
        all_fields[:, i] = (all_fields[:, i] - mean_values[i]) / std_values[i]

    if full_conditioning:
        all_nominal_full = np.zeros((dataset_size * stat, len(min_values), 128, 128), dtype=np.float32)
        for i in range(all_fields.shape[0]):
            for j in range(len(min_values)):
                all_nominal_full[i, j, :, :] = all_nominal[i, j]

        all_nominal = all_nominal_full

    tensor_nn_in = torch.tensor(all_nominal, dtype=dtype)
    tensor_nn_ou = torch.tensor(all_fields[:, : len(mean_values)], dtype=dtype)
    tensor_nn_cs = torch.tensor(all_fields[:, len(mean_values) :], dtype=dtype)

    nn_dataset = TensorDataset(tensor_nn_in, tensor_nn_ou, tensor_nn_cs)
    return nn_dataset


def load_cpcf_test(full_conditioning: bool):
    image_size = 128
    dataset_size = 16

    f = h5py.File(f"dynamic_stall_test.h5", "r")

    all_fields = f["fields"][:].reshape(dataset_size * stat, 8, image_size, image_size)
    all_nominal = f["nominal_condition"][:].reshape(dataset_size * stat, 4)

    f.close()

    for i in range(len(min_values)):
        all_nominal[:, i] = (all_nominal[:, i] - min_values[i]) / (max_values[i] - min_values[i])

    for i in range(len(mean_values)):
        all_fields[:, i] = (all_fields[:, i] - mean_values[i]) / std_values[i]

    if full_conditioning:
        all_nominal_full = np.zeros((dataset_size * stat, len(min_values), 128, 128), dtype=np.float32)
        for i in range(all_fields.shape[0]):
            for j in range(len(min_values)):
                all_nominal_full[i, j, :, :] = all_nominal[i, j]

        all_nominal = all_nominal_full

    nn_cs = all_fields[:, len(mean_values) :].reshape(dataset_size, stat, 2, image_size, image_size).mean(axis=1)
    all_fields = all_fields[:, : len(mean_values)].reshape(dataset_size, stat, len(mean_values), image_size, image_size)
    nn_ou = np.zeros((dataset_size, 2, len(mean_values), image_size, image_size), dtype=np.float32)
    nn_ou[:, 0, :, :, :] = all_fields.mean(axis=1)
    nn_ou[:, 1, :, :, :] = all_fields.std(axis=1)

    tensor_nn_in = torch.tensor(all_nominal[::stat], dtype=dtype)
    tensor_nn_ou = torch.tensor(nn_ou, dtype=dtype)
    tensor_nn_cs = torch.tensor(nn_cs, dtype=dtype)

    nn_dataset = TensorDataset(tensor_nn_in, tensor_nn_ou, tensor_nn_cs)
    return nn_dataset
