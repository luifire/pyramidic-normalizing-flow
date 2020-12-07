import warnings
from torchviz.dot import make_dot
import numpy as np

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
        if DATA_SET == DataSet.CIFAR:
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
    dot.render("graph", format="svg", view=False)


def print_parameter_weights(pyrFlow):
    for layer in pyrFlow.layer_list:
        layer.print_parameter()
        #for i in range(KERNEL_SIZE_SQ):
        #    diag = pyrFlow.layer_list[0].bundle[i].weights.diagonal()
        #    d = diag.abs()
        #    print(f"{i} diag min: {d.min():.3e} max: {d.max():.3e} prod: {d.prod():.3e} avg total {diag.mean():.3e}")
            #printt("grad", pyrFlow.conv_1.bundle[0].weights)
