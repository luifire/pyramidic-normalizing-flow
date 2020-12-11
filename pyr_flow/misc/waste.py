"""
def cut_data(data):
  #printt("list", list(data.shape))
  #at = list(data.shape)[2] % KERNEL_SIZE
  at = data.shape[2] % KERNEL_SIZE
  return data[:, :, at:, at:]
"""


"""
früheres weight matrix erzeugung

#printt("weights", diag)
        #printt("diag", diag)
        normal_diag_matrix = torch.diag_embed(diag)
        #printt("normal_diag_matrix", normal_diag_matrix)

        zero_diag_matrix = weights - normal_diag_matrix
        #print("zero_diag_matrix", zero_diag_matrix)

        # exp(diag)+1
        #exped_diag = torch.exp(diag) + 1
        warn("currently, diag could become 0")
        exped_diag = diag
        # printt("exped_diag", exped_diag)
        conv_matrix = zero_diag_matrix + torch.diag_embed(exped_diag)
        #printt("conv_matrix", conv_matrix)

        # printt("x", x.shape)
        conv_matrix = conv_matrix#.unsqueeze(0).unsqueeze(0)
        #printt("conv_matrix", conv_matrix.shape)

        det = torch.prod(exped_diag)

        if torch.isnan(det).any():
            print("asdf")

        assert det.item() != 0
        #printt("det", det.item())
        #printt("dev ende", conv_matrix.device)

        return conv_matrix, det



"""


""""
erste conv action 
        # x.shape == batch, channel, width, height
        # do the convolution myself
        # first to the right and then down
        # x.narrow(input, dim, start, length)
        #printt("x", x)
        #duplicate = torch.zeros_like(x)
        new_x = torch.empty_like(x)
        for h in range(height // KERNEL_SIZE):
            row = x.narrow(HEIGHT_DIM, h*KERNEL_SIZE, KERNEL_SIZE)
            for w in range(width // KERNEL_SIZE):
                patch = row.narrow(WIDTH_DIM, w*KERNEL_SIZE, KERNEL_SIZE)
                #printt("patch", patch)
                flattened = patch.reshape((batch_size, channel_count, self.kernel_size_sq))

                new_patch = torch.matmul(flattened, conv_matrix)

                #unflattened =
                new_x[:,:, h*KERNEL_SIZE:(h+1)*KERNEL_SIZE, w*KERNEL_SIZE:(w+1)*KERNEL_SIZE] = new_patch
                #printt("flattened", flattened)
                #rint(str(w) + "-" + str(h))
                #print(patch.shape)
                #print(patch)
                #duplicate[:,:,h*KERNEL_SIZE:(h+1)*KERNEL_SIZE, w*KERNEL_SIZE:(w+1)*KERNEL_SIZE] = patch
                #exit(1)
            #print(duplicate)
            #exit(1)

        #a = x - duplicate
        #printt("sum", a.sum())
        # amount of convs fits

        new_x = new_x * norm
        return new_x
        #exit(1)

        # printt("norm", norm)
        #x = x.view(-1, self.kernel_size*self.kernel_size, x.shape[2] // KERNEL_SIZE, x.shape[3] // KERNEL_SIZE)

        #y = F.conv2d(x, conv_matrix, stride=self.kernel_size)
        #print("y", y.shape)
        #normalization term
        #y = y / norm

        #print(y.requires_grad)
        #return y
        return None
        """

import torch.nn as nn

from pyr_flow.constants import *
from pyr_flow.model.layer_module import LayerModule


class InvertiblePolynome(LayerModule):

    def __init__(self):
        super().__init__()

        self.hole_filler_vector = None
        self.holed_shift_matrix = None
        self.dummy = nn.Parameter(torch.zeros(1, device=DEVICE))

    def _get_holed_shift_matrix_and_filler_vector(self, total_dim):
        if self.holed_shift_matrix is not None and self.hole_filler_vector is not None:
            return self.holed_shift_matrix, self.hole_filler_vector

        shift_matrix = torch.zeros(total_dim, total_dim, device=DEVICE)
        hole_filler_matrix = torch.zeros(total_dim, total_dim, device=DEVICE)

        for i in range(total_dim):
            a = 1 * ((i + 1) % 2)

            if i+1 < total_dim:
                shift_matrix[i, i + 1] = a
            hole_filler_matrix[i, i] = a

        zero_hole_filler = torch.ones(total_dim, device=DEVICE)
        hole_filler = zero_hole_filler.matmul(hole_filler_matrix)

        # shift_matrix is like
        # [[0., 1., 0., 0., 0., 0.,
        # [0., 0., 0., 0., 0., 0.,
        # [0., 0., 0., 1., 0., 0.,
        # [0., 0., 0., 0., 0., 0.,

        # hole filler is like
        # [1, 0, 1, 0, ...]
        self.holed_shift_matrix = shift_matrix
        self.hole_filler_vector = hole_filler

        #printt("shift_matrix", shift_matrix)
        #printt("hole_filler", hole_filler)
        return self.holed_shift_matrix, self.hole_filler_vector

    def forward(self, x: torch.Tensor):
        # an example for why this is correct is in misc.test_polynomes
        original_shape = x.shape
        #flat = x.flatten(1, -1)
        #total_dim = flat.shape[1]
        total_dim = original_shape[CHANNEL_DIM]

        holed_shift_matrix, hole_filler_vector = self._get_holed_shift_matrix_and_filler_vector(total_dim)

        holed = x.matmul(holed_shift_matrix)
        multiplier = holed + hole_filler_vector
        multiplier[multiplier==0] = 1
        #printt("x", x[0, 1:2])
        x = multiplier * x

        #printt("holed", holed[0,1:2])
        #printt("result", multiplier[0, 1:2])
        #printt("x danach", x[0, 1:2])

        # sum log |x|
        logd_det = multiplier.abs().log().sum(1).sum(1).sum(1)

        x = x.reshape(original_shape)

        return x, logd_det

"""
        shift_matrix = torch.zeros(image_depth, image_depth, device=DEVICE)
        for i in range(image_depth):
            shift_matrix[i, (i+1) % image_depth] = 1

        if jump_over_pixel:
            # for images with more channels we want to change the entire pixel
            for _ in range(PIXEL_DEPTH - 1): # ** CHANNEL_COUNT didn't work
                shift_matrix = shift_matrix.matmul(shift_matrix)
        """
"""
if False and diag.prod().item() == 0:
    print(self.name)
    print(diag)

    print("diag has zeros")
    """


"""
            step = step.reshape((batch_size, -1))
            n_rv = step.shape[1]
            prior = self._get_prior(n_rv)
            prior_ll = prior.log_prob(step)

            lnorm_map = pyramid_steps_lnorm[i]
            lnorm = lnorm_map.reshape((batch_size, -1)).sum(1)
            # -log(k) * rv_size um die Discretisierung weg zu rechnen
            discretisation_offset = self.log_k * n_rv

            corrected_prior_ll = prior_ll - discretisation_offset

            ll = corrected_prior_ll + lnorm
            #ll = corrected_prior_ll #+ lnorm
            #ll = summed_logd_det
            nll = -ll.mean()

            # pyramid weighting
            weight = 1 / self.weight_decrease**i
            loss += nll * weight

            unweighted_prior += -corrected_prior_ll
            unweighted_lnorm += -lnorm
            #if loss.isnan() or loss.isinf():
            #    warn("loss is " + loss.item())
            #    exit(1)

        return loss, unweighted_prior.mean(), unweighted_lnorm.mean()
        """
