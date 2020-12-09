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
        prior_avg = 0
        norm_avg = 0

        for batch_idx, (data, _) in enumerate(self.test_loader):
            pyramid_steps, norm = pyrFlow(data)
            loss, prior = pyramid_loss(pyramid_steps, norm)
            loss_avg += loss.item()
            prior_avg += -prior.mean().item()
            norm_avg += -norm.mean().item()

        # make average (/N) and make bits per dim
        loss_norming = len(self.test_loader) * BITS_PER_DIM_NORM
        loss_avg /= loss_norming
        prior_avg /= loss_norming
        norm_avg /= loss_norming
        print('*******************')
        print(f'Evaluation loss: {loss_avg:.5f} Prior: {prior_avg:.5f} Norm: {norm_avg:.5f}')
        print('*******************')

        return loss_avg

