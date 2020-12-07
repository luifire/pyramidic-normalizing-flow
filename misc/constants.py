import torch
from enum import Enum

DEVICE = torch.device("cuda:0")

KERNEL_SIZE = 3
#print("!!!!!!!!!! Kernel Size 2")
KERNEL_SIZE_SQ = KERNEL_SIZE**2

CHANNEL_DIM = 1
HEIGHT_DIM = 2
WIDTH_DIM = 3

n_epochs = 500
batch_size_train = 32
batch_size_test = 1000
learning_rate = 1e-4
# z.t. 10-5 für GLOW etc.
#learning_rate = 5e-2
#momentum = 0.5
weight_decay = 5e-5
log_interval = 100

random_seed = 1337
torch.manual_seed(random_seed)

torch.backends.cudnn.deterministic = True
torch.set_printoptions(linewidth=130)


class DataSet(Enum):
    MNIST = 1
    CIFAR = 2
    SVHN = 3


DATA_SET = DataSet.CIFAR

DATA_WIDTH, DATA_HEIGHT, CHANNEL_COUNT = None, None, None

if DATA_SET == DataSet.MNIST:
    DATA_WIDTH = 30
    DATA_HEIGHT = 30
    CHANNEL_COUNT = 1
elif DATA_SET == DataSet.CIFAR:
    DATA_WIDTH = 33
    DATA_HEIGHT = 33
    CHANNEL_COUNT = 3
else:
    raise Exception("uncovered")


def _printt(name, var):
    print(name + ": " + str(var))

_printt("learning_rate", learning_rate)
_printt("batch_size_train", batch_size_train)
_printt("weight_decay", weight_decay)
_printt("Dataset", DATA_SET)
_printt("Device", DEVICE)