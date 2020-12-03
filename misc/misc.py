
def printt(name, var):
    print(name)
    shape = "Shape: " + str(list(var.shape)) if "shape" in dir(var) else ""
    print(str(var) + "\t" + shape)