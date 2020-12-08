import warnings
from torchviz.dot import make_dot
import numpy as np
import os
from shutil import rmtree

from misc.constants import *

Batch_Idx = 0

def _shape_str(var):
    return str(list(var.shape))


def printt(name, var):
    print(name)
    shape = "Shape: " + _shape_str(var) if "shape" in dir(var) else ""
    print(str(var) + "\t" + shape)


def prints(name, var):
    print(name + " shape: " + _shape_str(var))


def warn(text):
    warnings.warn(text)


def plot_data(data):
    import matplotlib.pyplot as plt
    import time

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        d = data[i].cpu().numpy()
        # unnormalize
        if TRAIN_DATA_SET == DataSet.CIFAR:
            #d = d / 2 + 0.5
            pass
        #print(d.shape)
        d = np.transpose(d, (1, 2, 0))
        plt.imshow(d)
        #plt.title("Ground Truth: {}".format(plot_mnist[i].cpu()))
        plt.xticks([])
        plt.yticks([])

        print(d.min().item())
        print(d.max().item())
    plt.show()
    exit(1)


def visualize(node, network):
    dot = make_dot(node, params=dict(network.named_parameters()))
    dot.render("../graph", format="svg", view=False)


def print_parameter_weights(pyrFlow):
    for layer in pyrFlow.layer_list:
        layer.print_parameter()
        #for i in range(KERNEL_SIZE_SQ):
        #    diag = pyrFlow.layer_list[0].bundle[i].weights.diagonal()
        #    d = diag.abs()
        #    print(f"{i} diag min: {d.min():.3e} max: {d.max():.3e} prod: {d.prod():.3e} avg total {diag.mean():.3e}")
            #printt("grad", pyrFlow.conv_1.bundle[0].weights)


def test_polynomes():
    a = torch.arange(2*2*2*3).float().reshape((2, 2, 2, 3)) + 1
    print(a)
    flat = a.flatten(1, -1)
    print(flat)
    print(flat.reshape(2,2,2,3))
    total_dim = flat.shape[1]


    shift_matrix = torch.zeros(total_dim, total_dim)
    hole_filler_matrix = torch.zeros(total_dim, total_dim)

    for i in range(total_dim):
        shift_matrix[i, (i + 1) % total_dim] = 1 * ((i+1) % 2)
        hole_filler_matrix[i, i] = 1 * ((i+1) % 2)

    rotated = flat.matmul(shift_matrix)
    print(rotated)

    print(shift_matrix)

    zero_hole_filler = torch.ones(total_dim)
    hole_filler = zero_hole_filler.matmul(hole_filler_matrix)
    print(hole_filler)

    result = rotated + hole_filler
    final = result * flat
    for a in range(2):
        print("--------")
        print(flat[a])
        print(result[a])
        print(final[a])

    exit(1)

#test_polynomes()


def clean_up_dir():
    if os.path.exists(STATE_DIR):
        rmtree(STATE_DIR)
    if not os.path.exists(STATE_DIR):
        os.makedirs(STATE_DIR)
