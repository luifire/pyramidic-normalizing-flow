import torchvision

from pyr_flow.constants import *



def load_dataset(dataset, batch_size_test=BATCH_SIZE_TEST, batch_size_train=BATCH_SIZE_TRAIN) \
        -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):

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

        data_train = torchvision.datasets.MNIST('/files/', train=True, download=DOWNLOAD, transform=transformation)
        data_test = torchvision.datasets.MNIST('/files/', train=False, download=DOWNLOAD, transform=transformation)
    elif dataset == DataSet.CIFAR:

        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # left, top, right and bottom
            # 32x32
            # torchvision.transforms.Pad([0, 0, 1, 1], padding_mode='edge'),
            torchvision.transforms.CenterCrop(CENTER_CROP)
        ])

        data_train = torchvision.datasets.CIFAR10('/files/', train=True, download=DOWNLOAD, transform=transformation)
        data_test = torchvision.datasets.CIFAR10('/files/', train=False, download=DOWNLOAD, transform=transformation)
    elif dataset == DataSet.SVHN:
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.CenterCrop(CENTER_CROP)
        ])

        data_train = torchvision.datasets.SVHN('/files/', download=DOWNLOAD, transform=transformation)
        data_test = None # there is no test, train split
    else:
        raise Exception("Ups!")

    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=batch_size_train,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=batch_size_test,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)

    return train_loader, test_loader

