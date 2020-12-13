import torch
from pyr_flow.constants import *
from pyr_flow.misc.misc import *

class Evaluation():

    def __init__(self, test_loader, eval_loader):
        self.test_loader = test_loader
        self.eval_loader = eval_loader

        #train, test = load_dataset(dataset)

        #print(dir(train))

        #self.load_dataset

    def evaluation(pyrFlow):
        pass

    def _eval_on_set(self, pyrFlow, pyramid_loss, set):
        loss_avg = 0
        ll_avg = 0
        norm_avg = 0
        top_avg = 0

        for batch_idx, (data, _) in enumerate(set):
            if batch_idx * BATCH_SIZE_TEST > 1000:
                break
            with torch.no_grad():
                pyramid_steps, lnorm_map = pyrFlow(data)
                loss, ll, unweighted_lnorm, top_ls = pyramid_loss(pyramid_steps, lnorm_map)
            loss_avg += loss.item()
            ll_avg += ll.mean().item()
            norm_avg += unweighted_lnorm.mean().item()
            top_avg += top_ls.mean().item()

        # make average (/N) and make bits per dim
        batch_size = len(self.test_loader)
        loss_avg /= batch_size
        ll_avg /= batch_size * BITS_PER_DIM_NORM
        norm_avg /= batch_size * BITS_PER_DIM_NORM

        output = f'loss {loss_avg:.5f} nll {ll_avg:.5f} Norm {norm_avg:.5f} Top Loss {top_avg:.5f}'
        print(output)
        return output

    def eval_on_normal_test_set(self, pyrFlow, pyramid_loss):

        print_separator()
        print('Evaluation')
        print('Test Set')
        a = self._eval_on_set(pyrFlow, pyramid_loss, self.test_loader)

        print('Eval Set')
        b = self._eval_on_set(pyrFlow, pyramid_loss, self.eval_loader)
        print_separator()

        return a, b

