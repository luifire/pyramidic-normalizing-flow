import torch
import numpy as np
from enum import Enum

DEVICE = torch.device("cuda:0")

N_EPOCHS = 500
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 2000
#LEARNING_RATE = 1e-4
LEARNING_RATE = 1e-5
#LEARNING_RATE = 1e-3
# z.t. 10-5 für GLOW etc.
WEIGHT_DECAY = 5e-5
LOG_INTERVAL = 100
EVAL_INTERVAL = 5 # every epoches
MAX_GRAD_NORM = 100

S_LOG_GATE_ALPHA_INIT = 0.1

PYRAMID_STEP_WEIGHTING = 1.5 # weighted by 1/2**i
LAST_PIXEL_BREAK_DOWN = 3 # last pixel consists of N pixels, then our pyramid steps will be /3

# KERNEL Stuff
INITIAL_KERNEL_SIZE = 3
INITIAL_KERNEL_SIZE_SQ = INITIAL_KERNEL_SIZE ** 2
COMBINE_NEIGHBOR_KERNEL_SIZE = 2
COMBINE_NEIGHBOR_KERNEL_SIZE_SQ = COMBINE_NEIGHBOR_KERNEL_SIZE ** 2

# Dimension Index
BATCH_DIM = 0
HEIGHT_DIM = 1
WIDTH_DIM = 2
CHANNEL_DIM = 3

# Torch Init
RANDOM_SEED = 1337
torch.manual_seed(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.set_printoptions(linewidth=130)
#torch.set_default_tensor_type(torch.DoubleTensor)

# Data Set
class DataSet(Enum):
    MNIST = 1
    CIFAR = 2
    SVHN = 3


TRAIN_DATA_SET = DataSet.CIFAR
#TRAIN_DATA_SET = DataSet.MNIST
#TEST_DATA_SET = DataSet.SVHN
DOWNLOAD = False


STATE_DIR = '../model_state/'
TRAIN_STATE_DIR = f'{STATE_DIR}{TRAIN_DATA_SET.name}/'

#DATA_WIDTH, DATA_HEIGHT, PIXEL_DEPTH, CENTER_CROP = None, None, None,

if TRAIN_DATA_SET == DataSet.MNIST:
    DATA_WIDTH = 30
    DATA_HEIGHT = 30
    PIXEL_DEPTH = 1
elif TRAIN_DATA_SET == DataSet.CIFAR:
    #DATA_WIDTH = 33
    #DATA_HEIGHT = 33
    PIXEL_DEPTH = 3

    CENTER_CROP = 24
else:
    raise Exception("uncovered")

if CENTER_CROP is not None:
    DATA_WIDTH, DATA_HEIGHT = CENTER_CROP, CENTER_CROP

TOTAL_IMAGE_DIMENSION = DATA_WIDTH * DATA_HEIGHT * PIXEL_DEPTH

# from Kirichenko
BITS_PER_DIM_NORM = np.log(2) * TOTAL_IMAGE_DIMENSION
