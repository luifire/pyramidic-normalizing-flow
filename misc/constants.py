import torch

DEVICE = torch.device("cuda:0")

KERNEL_SIZE = 3
KERNEL_SIZE_SQ = KERNEL_SIZE**2

HEIGHT_DIM = 2
WIDTH_DIM = 3