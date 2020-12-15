import torch.nn as nn

from pyr_flow.model.reshaping.shift_channel_layer import ChannelShifter
from pyr_flow.model.layer_module import LayerModule

from pyr_flow.constants import *
from pyr_flow.model.computation_layers.depth_conv_layer import DepthConv
from pyr_flow.model.computation_layers.depth_conv_2 import DepthConv2


class DepthConvBundle(LayerModule):
    s_bundle_number = 1

    """ jump_over_pixels : if true, then for rotation we rotate over them (in channel shifter)"""
    """ if bundle_size <= 0, it will be ignored and the complete thing will be iterated
    conv_type == 1 : standard
    conv_type == 2 : only first line is learnable
    """

    def __init__(self, total_pixel_depth, internal_pixel_depth, jump_over_pixels, bundle_size=-1, conv_type=1):
        super().__init__()

        self.bundle = nn.ModuleList()
        self.name = str(DepthConvBundle.s_bundle_number)
        DepthConvBundle.s_bundle_number += 1
        self.channel_shifter = ChannelShifter(total_pixel_depth=total_pixel_depth,
                                              internal_pixel_depth=internal_pixel_depth,
                                              jump_over_pixels=jump_over_pixels)

        bias = torch.normal(mean=0, std=0.5, size=[total_pixel_depth], device=DEVICE)
        self.bias = nn.Parameter(bias, requires_grad=True)  # nn.Parameter is a Tensor that's a module parameter.

        true_bundle_size = self._compute_actual_bundle_size(bundle_size=bundle_size,
                                                            total_pixel_depth=total_pixel_depth,
                                                            internal_pixel_depth=internal_pixel_depth,
                                                            jump_over_pixels=jump_over_pixels)
        # print(f'total_pixel_depth {total_pixel_depth} internal_pixel_depth {internal_pixel_depth}')
        for i in range(true_bundle_size):
            conv_name = "Bundle " + self.name + " Conv " + str(i)
            if conv_type == 1:
                self.bundle.append(DepthConv(name=conv_name,
                                             total_pixel_depth=total_pixel_depth,
                                             internal_pixel_depth=internal_pixel_depth,
                                             jump_over_pixels=jump_over_pixels,
                                             pixel_idx=i))
            elif conv_type == 2:
                self.bundle.append(DepthConv2(name=conv_name,
                                              total_pixel_depth=total_pixel_depth,
                                              internal_pixel_depth=internal_pixel_depth))
            else:
                raise Exception('no proper conv_type')

        print(f'Conv Bundle {self.name} - Size: {true_bundle_size} total_pixel_depth: {total_pixel_depth} '
              f'internal_pixel_depth: {internal_pixel_depth} jump_over_pixels :{jump_over_pixels} |'
              f'param count: {self.get_parameter_count()}')

    @staticmethod
    def _compute_actual_bundle_size(bundle_size, total_pixel_depth, internal_pixel_depth, jump_over_pixels):
        if bundle_size <= 0:
            if jump_over_pixels:
                res = total_pixel_depth // internal_pixel_depth
            else:
                res = total_pixel_depth
        else:
            res = bundle_size
        return res

    def __call__(self, x, lnorm_map):
        for i, layer in enumerate(self.bundle):
            x, lnorm_map = layer(x, lnorm_map)
            x = self.channel_shifter(x)
            lnorm_map = self.channel_shifter(lnorm_map)

        return x, lnorm_map

    def print_parameter(self):
        for layer in self.bundle:
            layer.print_parameter()
        print(f"bias mean {self.bias.mean(): .3e}")

    def get_parameter_count(self):
        return sum([layer.get_parameter_count() for layer in self.bundle]) + self.bias.shape[0]
