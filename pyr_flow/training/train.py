import torch.optim as optim

from model.pyramid_flow_model import PyramidFlowModel
from model.computation_layers.pyramid_loss import PyramidLoss

from misc.misc import *
from constants import *
from utils import data_utils, training_utils
from training.evaluation import Evaluation


print_constants()

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

printt("Param Count: ", pyrFlow.get_parameter_count())
print_separator()

def train(epoch):
    pyrFlow.train()
    global Batch_Idx
    for batch_idx, (data, target) in enumerate(train_loader):
        Batch_Idx += 1
        data = data.to(DEVICE)

        #plot_data(data)
        #print(batch_idx)

        optimizer.zero_grad()

        pyramid_steps, pyramid_steps_lnorm = pyrFlow(data)
        loss, nll, unweighted_lnorm, top_nll = pyramid_loss(pyramid_steps, pyramid_steps_lnorm)

        loss.backward()
        #if batch_idx > 881 and False:
        #    printt("loss", loss)
        #    printt("grad", pyrFlow.conv_1.bundle[0].weights.grad)

        # TODO: clip gradient
        training_utils.clip_grad_norm(optimizer, MAX_GRAD_NORM)

        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            if epoch == 0 and batch_idx == 0:
                try:
                    create_flow_graph(loss, pyrFlow)
                except RecursionError:
                    warn("Couldn't draw graph. Too many recursions")
                PyramidLoss.print_loss_size(pyramid_steps)

            #printt(str(batch_idx), pyrFlow.parameters())
            individual_loss = f" nll: {nll.item() / BITS_PER_DIM_NORM:.5f} " \
                              f" norm: {unweighted_lnorm.item() / BITS_PER_DIM_NORM:.5f} " \
                              f" top nll: {top_nll.item():.5f}"
            percentage = 100. * batch_idx / len(train_loader)
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({percentage:.0f}%)]'
                  f'\tweighted Loss: {loss.item():.5f} bits/dim:' + individual_loss)

            if batch_idx % (LOG_INTERVAL*3) == 0:
                max_val = max([steps.abs().max() for steps in pyramid_steps])
                min_val = min([steps.abs().min() for steps in pyramid_steps])
                print(f'Prior Domain Max {max_val:.3f} Min {min_val:.3f}')
                # TODO check std dev
            #train_losses.append(loss.item())
            #train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

for epoch in range(N_EPOCHS):
    if (epoch + 1) % EVAL_INTERVAL == 0:
        eval_loss = evaluation.eval_on_normal_test_set(pyrFlow, pyramid_loss, TOTAL_IMAGE_DIMENSION)
        name = f'{epoch} - loss - {eval_loss:.5f}'

        torch.save(pyrFlow.state_dict(), f'{STATE_DIR}/{name}.model')
        torch.save(optimizer.state_dict(), f'{STATE_DIR}/{name}.optimizer')

    train(epoch)
    pyrFlow.print_parameter()
