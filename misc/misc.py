import warnings

Batch_Idx = 0

def _shape_str(var):
    return str(list(var.shape))


def printt(name, var):
    print(name)
    shape = "Shape: " + _shape_str(var) if "shape" in dir(var) else ""
    print(str(var) + "\t" + shape)


def print_shape(name, var):
    print(name + " shape: " + _shape_str(var))


def warn(text):
    warnings.warn(text)


def plot_mnist(data):
    import matplotlib.pyplot as plt
    import time

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(data[i][0].cpu(), cmap='gray', interpolation='none')
        #plt.title("Ground Truth: {}".format(plot_mnist[i].cpu()))
        plt.xticks([])
        plt.yticks([])
        plt.show()
        while True:
            time.sleep(1)
        exit(1)
