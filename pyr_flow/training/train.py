import torch.optim as optim

from pyr_flow.model.pyramid_flow_model import PyramidFlowModel
from pyr_flow.model.pyramid_loss import PyramidLoss

from pyr_flow.utils import training_utils, data_utils
from pyr_flow.training.evaluation import Evaluation

from pyr_flow.misc.misc import *

if __name__=='__main__':

    print_constants()

    # DATA
    train_loader, test_loader = data_utils.load_dataset(TRAIN_DATA_SET)
    eval_loader, _ = data_utils.load_dataset(DataSet.SVHN)
    # EVALUATION
    evaluation = Evaluation(test_loader, eval_loader)

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

            # randomly add noise (for time efficiency)
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    noise = torch.normal(mean=0, std=0.0001, size=data.shape, device=DEVICE)
                    data += noise

            #if batch_idx > 100:
            #    exit(0)
            #plot_data(data)
            #print(batch_idx)

            optimizer.zero_grad()

            pyramid_steps, pyramid_steps_lnorm = pyrFlow(data)
            #use_norm = batch_idx % 100 == 0 # only use norm every x time as the gradient towards the norm is way bigger
            use_norm = True
            loss, nll, unweighted_lnorm, top_nll = pyramid_loss(pyramid_steps, pyramid_steps_lnorm, use_norm)

            loss.backward()

            training_utils.clip_grad_norm(optimizer, MAX_GRAD_NORM)

            optimizer.step()

            #pyrFlow.print_parameter()
            if batch_idx % LOG_INTERVAL == 0:
                if epoch == 0 and batch_idx == 0:
                    PyramidLoss.print_loss_size(pyramid_steps)

                    try:
                        create_flow_graph(loss, pyrFlow)
                    except RecursionError:
                        warn("Couldn't draw graph. Too many recursions")

                nll = nll.mean()
                unweighted_lnorm = unweighted_lnorm.mean()
                top_nll = top_nll.mean()

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
                    avg_val = np.mean([steps.abs().mean().item() for steps in pyramid_steps])
                    print(f'Prior Domain Max {max_val:.5f} Min {min_val:.5f} Avg {avg_val:.5f}')

                if torch.isnan(loss).any():
                    exit(2)

        return nll.mean().item() / BITS_PER_DIM_NORM, top_nll.mean().item()
                # TODO check std dev
                #train_losses.append(loss.item())
                #train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))


    for epoch in range(N_EPOCHS):
        #a, b = evaluation.eval_on_normal_test_set(pyrFlow, pyramid_loss)

        last_nll, last_top_nll = train(epoch)
        pyrFlow.print_parameter()

        if (epoch + 1) % EVAL_INTERVAL == 0:
            a, b = evaluation.eval_on_normal_test_set(pyrFlow, pyramid_loss)
            name = f'{epoch} - Test {a} Eval {b}'
        else:
            name = f'{epoch} - loss - {last_nll:.5f} top_nll {last_top_nll:.4f}'


        torch.save(pyrFlow.state_dict(), f'{TRAIN_STATE_DIR}/{name}.model')
        torch.save(optimizer.state_dict(), f'{TRAIN_STATE_DIR}/{name}.optimizer')
