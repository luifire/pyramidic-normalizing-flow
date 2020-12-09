import torchvision

from misc.constants import *



def load_dataset(dataset) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):

    # TODO add noise to data.
    # TODO: Note that all data is not normalized!

    # from NICE:
    # As prescribed in (Uria et al., 2013), we use a dequantized version of the data:
    # we add a uniform noise of 1 / 256 to the data and rescale it to be in [0, 1]D after dequantization. We add
    # a uniform noise of 1/128 and rescale the data to be in [−1, 1]D for CIFAR-10.

    if dataset == DataSet.MNIST:
        # 28x28
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # left, top, right and bottom
            # for MNIST
            torchvision.transforms.Pad([0, 0, 2, 2], padding_mode='edge'),
        ])

        data_train = torchvision.datasets.MNIST('/files/', train=True, download=True, transform=transformation)
        data_test = torchvision.datasets.MNIST('/files/', train=False, download=True, transform=transformation)
    elif dataset == DataSet.CIFAR:
        #32x32
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # left, top, right and bottom
            torchvision.transforms.Pad([0, 0, 1, 1], padding_mode='edge'),
        ])

        data_train = torchvision.datasets.CIFAR10('/files/', train=True, download=True, transform=transformation)
        data_test = torchvision.datasets.CIFAR10('/files/', train=False, download=True, transform=transformation)
    else:
        raise Exception("Ups!")

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE_TEST, shuffle=True)

    return train_loader, test_loader

