from pyr_flow.constants import *


def get_shift_matrix(size, shift_by=1):
    shift_by_one = torch.zeros(size, size, device=DEVICE)
    for i in range(size):
        shift_by_one[i, (i + 1) % size] = 1

    shift_by_n = torch.eye(size, device=DEVICE)
    for _ in range(shift_by):
        shift_by_n = shift_by_n.matmul(shift_by_one)

    return shift_by_n

def channel_to_last_dim(x):
    return x.permute(0, 2, 3, 1)


def __channel_normal_position(x):
    return x.permute(0, 3, 1, 2)
