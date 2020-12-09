import torch.optim as optim
import os

from model.pyramid_flow_model import PyramidFlowModel
from model.pyramid_loss import PyramidLoss

from misc.misc import *
from misc.constants import *
from utils import data_utils, training_utils
from training.evaluation import Evaluation

# DATA
train_loader, test_loader = data_utils.load_dataset(TRAIN_DATA_SET)
# EVALUATION
evaluation = Evaluation(test_loader)

# MODEL
pyrFlow = PyramidFlowModel()
pyramid_loss = PyramidLoss(DATA_WIDTH * DATA_HEIGHT * PIXEL_DEPTH, k=256)
# OPTIMIZER
optimizer = optim.Adam(pyrFlow.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# FOLDER INIT
clean_up_dir()

print("Currently Paramcount incorrect (it's less params)")
printt("Param Count: ", pyrFlow.get_parameter_count())
warn("check k=256")


def train(epoch):
    pyrFlow.train()
    global Batch_Idx
    for batch_idx, (data, target) in enumerate(train_loader):
        Batch_Idx = batch_idx
        data = data.to(DEVICE)

        #plot_data(data)
        #print(batch_idx)

        optimizer.zero_grad()

        pyramid_steps, norm = pyrFlow(data)
        loss, prior = pyramid_loss(pyramid_steps, norm)

        loss.backward()
        #if batch_idx > 881 and False:
        #    printt("loss", loss)
        #    printt("grad", pyrFlow.conv_1.bundle[0].weights.grad)

        # TODO: clip gradient
        training_utils.clip_grad_norm(optimizer, MAX_GRAD_NORM)

        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            if batch_idx == 0:
                visualize(loss, pyrFlow)

            #printt(str(batch_idx), pyrFlow.parameters())
            individual_loss = f" Prior: {-prior.mean().item() / BITS_PER_DIM_NORM:.3f} " \
                              f"norm: {-norm.mean().item() / BITS_PER_DIM_NORM:.3f}"
            percentage = 100. * batch_idx / len(train_loader)
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({percentage:.0f}%)]'
                  f'\tLoss: {loss.item() / BITS_PER_DIM_NORM:.3f} bits/dim' + individual_loss)

            if batch_idx % (LOG_INTERVAL*3) == 0:
                max_val = max([steps.abs().max() for steps in pyramid_steps])
                min_val = min([steps.abs().min() for steps in pyramid_steps])
                print(f'Prior Domain Max {max_val:.3f} Min {min_val:.3f}')
                # TODO check std dev
            #train_losses.append(loss.item())
            #train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

for epoch in range(1, N_EPOCHS + 1):
    pyrFlow.print_parameter()
    if epoch % EVAL_INTERVAL == 0:
        eval_loss = evaluation.eval_on_normal_test_set(pyrFlow, pyramid_loss, TOTAL_IMAGE_DIMENSION)
        name = f'{epoch} - loss - {eval_loss:.3f}'

        torch.save(pyrFlow.state_dict(), f'{STATE_DIR}/{name}.model')
        torch.save(optimizer.state_dict(), f'{STATE_DIR}/{name}.optimizer')

    train(epoch)
