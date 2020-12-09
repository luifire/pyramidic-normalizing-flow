import torch.nn as nn


class LayerModule(nn.Module):

    def print_parameter(self):
        pass

    # as depth_conv_layer contains more parameter then it actually has,
    # we implemented a function to compute the true param count
    def get_parameter_count(self):
        return 0

    def forward(self, x, log_norm_map):
        raise Exception('mind the map')