import warnings


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