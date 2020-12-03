import torch.nn as nn
import torch
import torch.nn.functional as F

from misc.misc import *
from misc.constants import *
# BIAS

class DepthConv(nn.Module):
    """  """
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_size_sq = kernel_size**2

        # 1.5 kind of suggested by IFA-VAE
        weights = torch.normal(mean=1.5, std=0.5, size=[self.kernel_size_sq, self.kernel_size_sq], device=DEVICE)

        #printt("init weights", weights)
        self.weights = nn.Parameter(weights, requires_grad=True)  # nn.Parameter is a Tensor that's a module parameter.
        #bias = torch.Tensor(size_out)
        #self.bias = nn.Parameter(bias)

    def _prepare_weight_matrix_and_norm(self):
        # make triangular matrix
        # printt("forward 1. weights", self.weights)
        printt("device", self.weights.device)

        weights = torch.triu(self.weights)

        diag = torch.diagonal(weights)
        #printt("diag", diag)
        normal_diag_matrix = torch.diag_embed(diag)
        #printt("normal_diag_matrix", normal_diag_matrix)

        zero_diag_matrix = weights - normal_diag_matrix
        #print("zero_diag_matrix", zero_diag_matrix)

        # exp(diag)+1
        exped_diag = torch.exp(diag) + 1
        # printt("exped_diag", exped_diag)
        conv_matrix = zero_diag_matrix + torch.diag_embed(exped_diag)
        #printt("conv_matrix", conv_matrix)

        # printt("x", x.shape)
        conv_matrix = conv_matrix.unsqueeze(0).unsqueeze(0)
        #printt("conv_matrix", conv_matrix.shape)

        norm = torch.prod(exped_diag).abs()
        printt("dev ende", conv_matrix.device)
        return conv_matrix, norm

    def forward(self, x : torch.Tensor):

        conv_matrix, norm = self._prepare_weight_matrix_and_norm()

        print_shape("x", x)
        #batch_size, channel_count, height, width = x.shape

        permuted = x.permute(*(0, 2, 3, 1))
        printt("x", x.device)

        warn("this might not walk and hasn't been checked!!!")
        x = permuted.matmul(conv_matrix)
        x = x.permute(0, 3, 1, 2)

        return x * norm


class DepthConvBundl(nn.Module):

    def __init__(self, kernel_size=KERNEL_SIZE, bundle_count=KERNEL_SIZE):
        super().__init__()
        self.bundle = []
        for i in range(bundle_count):
            self.bundle.append(DepthConv(kernel_size))

    def __call__(self, x):
        for conv in self.bundle:
            x = conv(x)
        return x

        """"
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

