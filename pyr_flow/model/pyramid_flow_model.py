from model.depth_conv_layer import *
from model.initial_reshaping import initialReshaping
from model.s_log_gate import SLogGate
from model.layer_module import LayerModule
from model.cut_off_layer import CutOff
from model.conv_bundle import DepthConvBundle
from model.invertible_polynomes import InvertiblePolynome

from misc.constants import *


class PyramidFlowModel(LayerModule):

    def __init__(self):
        super().__init__()

        # TODO make this prettier
        self.layer_list = nn.ModuleList()
        channel_size = KERNEL_SIZE_SQ * PIXEL_DEPTH
        bundle_size = channel_size # s.t. every pixel consists of all others
        bundle_size = KERNEL_SIZE_SQ
        #bundle_size = 1
        #bundle_size = 2
        bundle_size_2 = KERNEL_SIZE_SQ
        #bundle_size_2 = 2
        #channel_size = 1 # warn
        self.layer_list.append(DepthConvBundle("1", channel_count=channel_size, bundle_size=bundle_size,
                                               jump_over_pixels=True))

        self.layer_list.append(InvertiblePolynome())

        self.layer_list.append(SLogGate())
        self.layer_list.append(DepthConvBundle("2", channel_count=channel_size, bundle_size=bundle_size_2,
                                               jump_over_pixels=True))
        self.layer_list.append(InvertiblePolynome())

        #self.layer_list.append(SLogGate())

        #self.layer_list.append(CutOff(channel_size // 2))
        #self.layer_list.append(DepthConvBundle("3", channel_count=channel_size, bundle_size=bundle_size,
        #                                   jump_over_pixels=True))
        
    def forward(self, x):
        pyramid_steps = []
        pyramid_steps_lnorm = []

        x = initialReshaping(x)
        lnorm_map = torch.zeros_like(x, device=DEVICE)
        log_norm = torch.zeros(x.shape[0], device=DEVICE)
        for layer in self.layer_list:
            if type(layer) is CutOff:
                x, cut_off = layer(x)
                pyramid_steps.append(cut_off)

                lnorm_map, cut_off_lnorm_map = layer(lnorm_map)
                pyramid_steps_lnorm.append(cut_off_lnorm_map)
            else: # normal procedure
                x, lnorm_map = layer(x, lnorm_map)
                #log_norm += x_log_norm
        # top data needs to be appended to the pyramid
        pyramid_steps.append(x)
        pyramid_steps_lnorm.append(lnorm_map)

        return pyramid_steps, pyramid_steps_lnorm

    def print_parameter(self):
        for layer in self.layer_list:
            layer.print_parameter()

    def get_parameter_count(self):
        return sum([layer.get_parameter_count() for layer in self.layer_list])
