import torch
import numpy as np
from enum import Enum

DEVICE = torch.device("cuda:0")

KERNEL_SIZE = 3
#print("!!!!!!!!!! Kernel Size 2")
KERNEL_SIZE_SQ = KERNEL_SIZE**2

BATCH_DIM = 0
HEIGHT_DIM = 1
WIDTH_DIM = 2
CHANNEL_DIM = 3

N_EPOCHS = 500
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 1e-4
#LEARNING_RATE = 1e-3
#learning_rate = 1e-1
# z.t. 10-5 für GLOW etc.
#learning_rate = 5e-2
#momentum = 0.5
WEIGHT_DECAY = 5e-5
LOG_INTERVAL = 100
EVAL_INTERVAL = 5 # every epoches
MAX_GRAD_NORM = 100

RANDOM_SEED = 1337
torch.manual_seed(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.set_printoptions(linewidth=130)
#torch.set_default_tensor_type(torch.DoubleTensor)


class DataSet(Enum):
    MNIST = 1
    CIFAR = 2
    SVHN = 3


TRAIN_DATA_SET = DataSet.CIFAR
TRAIN_DATA_SET = DataSet.MNIST
#TEST_DATA_SET = DataSet.SVHN
DOWNLOAD = False

STATE_DIR = f'../model_state/{TRAIN_DATA_SET.name}/'

DATA_WIDTH, DATA_HEIGHT, PIXEL_DEPTH = None, None, None

if TRAIN_DATA_SET == DataSet.MNIST:
    DATA_WIDTH = 30
    DATA_HEIGHT = 30
    PIXEL_DEPTH = 1
elif TRAIN_DATA_SET == DataSet.CIFAR:
    DATA_WIDTH = 33
    DATA_HEIGHT = 33
    PIXEL_DEPTH = 3
else:
    raise Exception("uncovered")

TOTAL_IMAGE_DIMENSION = DATA_WIDTH * DATA_HEIGHT * PIXEL_DEPTH

# from Kirichenko
BITS_PER_DIM_NORM = np.log(2) * TOTAL_IMAGE_DIMENSION

def _printt(name, var):
    print(name + ": " + str(var))


_printt("learning_rate", LEARNING_RATE)
_printt("batch_size_train", BATCH_SIZE_TRAIN)
_printt("weight_decay", WEIGHT_DECAY)
_printt("Dat Set", TRAIN_DATA_SET)
_printt("Data Size", DATA_WIDTH*DATA_HEIGHT)
_printt("Device", DEVICE)