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
        #printt("device", self.weights.device)

        weights = torch.triu(self.weights)

        diag = torch.diagonal(weights)
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

    def forward(self, x : torch.Tensor):

        conv_matrix, det = self._prepare_weight_matrix_and_norm()

        #print_shape("x", x)
        _, _, height, width = x.shape

        permuted = x.permute(*(0, 2, 3, 1))

        warn("achtung beim splitten! [0] bekommt die wenigste info und [8] die meiste!!!")
        x = permuted.matmul(conv_matrix)
        x = x.permute(0, 3, 1, 2)

        # TODO speedup
        amount_of_convolutions = (height // 3) * (width // 3)
        warn("this doesn't feel correct (had a bad morning)")
        # |det| == |det(Kernel)^amount_of_convolutions|
        norm = (det ** amount_of_convolutions).abs()

        if torch.isnan(norm).any():
            print("asdf")
        assert torch.isnan(norm).any() == False

        # log |det|^-1 = -log |det|
        loged_norm = -norm.log()
        warn("for the normalization, the -norm.log() part happens where the norm is created!")

""""
        Momentan ist folgendes Problem:
        der ich bekomme nach einer gewissen Anzahl an Runden nan in die diagonale
        jetzt versuche ich den Gradienten anzuschauen um zu kucken, ob es an dem
        liegt
        Ideen könnten sein, den Normalisierungsterm nicht für die Gradientenrechnung zu benutzen.
        printt("grad", self.weights.grad)
        if Batch_Idx >= 769:
            pass
"""
        return x, loged_norm


class DepthConvBundle(nn.Module):

    def __init__(self, kernel_size=KERNEL_SIZE, bundle_count=KERNEL_SIZE_SQ):
        super().__init__()
        #self.logd_norms = torch.empty(bundle_count)
        self.bundle = nn.ModuleList()
        for _ in range(bundle_count):
            self.bundle.append(DepthConv(kernel_size))

    def __call__(self, x):
        logd_norm_sums = torch.zeros(1, device=DEVICE)
        for i, conv in enumerate(self.bundle):
            x, log_norm = conv(x)
            logd_norm_sums = logd_norm_sums.add(log_norm)
        #logd_norm_sum = self.logd_norms.sum()
        #self.logd_norms
        return x, logd_norm_sums

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

