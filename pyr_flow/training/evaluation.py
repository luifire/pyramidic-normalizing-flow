from misc.constants import *

class Evaluation():

    def __init__(self, test_loader):
        self.test_loader = test_loader

        #train, test = load_dataset(dataset)

        #print(dir(train))

        #self.load_dataset

    def evaluation(pyrFlow):
        pass

    def eval_on_normal_test_set(self, pyrFlow, pyramid_loss, pixel_count):
        loss_avg = 0
        ll_avg = 0
        norm_avg = 0

        for batch_idx, (data, _) in enumerate(self.test_loader):
            pyramid_steps, lnorm_map = pyrFlow(data)
            loss, ll, unweighted_lnorm = pyramid_loss(pyramid_steps, lnorm_map)
            loss_avg += loss.item()
            ll_avg += ll.item()
            norm_avg += unweighted_lnorm.item()

        # make average (/N) and make bits per dim
        batch_size = len(self.test_loader)
        loss_avg /= batch_size
        ll_avg /= batch_size * BITS_PER_DIM_NORM
        norm_avg /= batch_size * BITS_PER_DIM_NORM
        print('*******************')
        print(f'Evaluation loss: {loss_avg:.5f} nll: {ll_avg:.5f} Norm: {norm_avg:.5f}')
        print('*******************')

        return loss_avg

