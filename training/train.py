import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

from model.pyramid_flow_model import PyramidFlowModel
from model.flow_loss import FlowLoss

from misc.misc import *
from misc.constants import *

#https://nextjournal.com/gkoehler/pytorch-mnist

n_epochs = 50
batch_size_train = 64
batch_size_test = 1000
warn("high learning rate")
learning_rate = 1e-3
#learning_rate = 5e-2
#momentum = 0.5
weight_decay = 5e-5
log_interval = 100



torch.backends.cudnn.deterministic = True

random_seed = 1337
torch.manual_seed(random_seed)

transformation = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                                #left, top, right and bottom
                                # for MNIST
                               torchvision.transforms.Pad([0,0,2,2], padding_mode='edge'),
                               #torchvision.transforms.Normalize(
                               #  (0.1307,), (0.3081,))
                                ])

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=transformation),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=transformation),
  batch_size=batch_size_test, shuffle=True)

pyrFlow = PyramidFlowModel()

warn("check k=256")
flow_loss = FlowLoss(DATA_WIDTH*DATA_HEIGHT, k=256)

#optimizer = optim.SGD(network.parameters(), lr=learning_rate,
#                      momentum=momentum)

optimizer = optim.Adam(pyrFlow.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

"""
def cut_data(data):
  #printt("list", list(data.shape))
  #at = list(data.shape)[2] % KERNEL_SIZE
  at = data.shape[2] % KERNEL_SIZE
  return data[:, :, at:, at:]
"""

def train(epoch):
    pyrFlow.train()
    global Batch_Idx
    for batch_idx, (data, target) in enumerate(train_loader):
        Batch_Idx = batch_idx
        data = data.to(DEVICE)
        #plot_mnist(data)
        print(batch_idx)
        #print_shape("data", data)
        #print(data.max().item())

        optimizer.zero_grad()

        y, norm = pyrFlow(data)
        loss = flow_loss(y, norm)

        if batch_idx > 768:
            printt("loss", loss)
            printt("grad", loss.grad)

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:

            #printt(str(batch_idx), pyrFlow.parameters())

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item() / (DATA_WIDTH*DATA_HEIGHT)))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            #torch.save(pyrFlow.state_dict(), './results/model.pth')
            #torch.save(optimizer.state_dict(), './results/optimizer.pth')


for epoch in range(1, n_epochs + 1):
    train(epoch)
