import torchvision
import torch.optim as optim

from model.pyramid_flow_model import PyramidFlowModel
from model.flow_loss import FlowLoss

from misc.misc import *
from misc.constants import *
from misc import helper
import time

#https://nextjournal.com/gkoehler/pytorch-mnist

train_loader, test_loader = helper.load_dataset()

pyrFlow = PyramidFlowModel()

warn("check k=256")
flow_loss = FlowLoss(DATA_WIDTH*DATA_HEIGHT*CHANNEL_COUNT, k=256)

optimizer = optim.Adam(pyrFlow.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
    pyrFlow.train()
    global Batch_Idx
    for batch_idx, (data, target) in enumerate(train_loader):
        Batch_Idx = batch_idx
        data = data.to(DEVICE)

        #plot_data(data)
        #print(batch_idx)

        optimizer.zero_grad()

        y, norm = pyrFlow(data)
        loss = flow_loss(y, norm)

        loss.backward()
        if batch_idx > 881 and False:
            printt("loss", loss)
            printt("grad", pyrFlow.conv_1.bundle[0].weights.grad)

        # TODO: clip gradient
        #utils.clip_grad_norm(optimizer, max_grad_norm)

        optimizer.step()

        if batch_idx % log_interval == 0:
            if batch_idx == 0:
                visualize(loss, pyrFlow)

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
    #print_parameter_weights(pyrFlow)
