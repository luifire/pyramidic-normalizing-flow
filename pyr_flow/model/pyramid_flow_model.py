from model.computation_layers.depth_conv_layer import *
from model.reshaping.initial_reshaping import merge_patches
from model.computation_layers.s_log_gate import SLogGate
from model.layer_module import LayerModule
from model.reshaping.cut_off_layer import CutOff
from model.computation_layers.conv_bundle import DepthConvBundle
from model.computation_layers.invertible_polynomes import InvertiblePolynome
from model.reshaping.combine_neighboring_info import CombineNeighbors

from utils.functional_utils import channel_to_last_dim

from constants import *


class PyramidFlowModel(LayerModule):

    def __init__(self):
        super().__init__()

        self.layer_list = nn.ModuleList()
        self.first_run_done = False
        internal_pixel_depth = PIXEL_DEPTH
        total_pixel_depth = INITIAL_KERNEL_SIZE_SQ * internal_pixel_depth

        #bundle_size = total_pixel_depth # s.t. every pixel consists of all others
        #bundle_size = INITIAL_KERNEL_SIZE
        #bundle_size = 1
        #bundle_size = 2
        #bundle_size_2 = INITIAL_KERNEL_SIZE
        #bundle_size_2 = 2
        #channel_size = 1 # warn
        """
        self.layer_list.append(DepthConvBundle("1", channel_count=channel_size, bundle_size=bundle_size,
                                               jump_over_pixels=True))

        self.layer_list.append(InvertiblePolynome())

        self.layer_list.append(SLogGate())
        self.layer_list.append(DepthConvBundle("2", channel_count=channel_size, bundle_size=bundle_size_2,
                                               jump_over_pixels=True))
        self.layer_list.append(InvertiblePolynome())
        """
        print_separator()
        print('Creating Pyramid Flow Network')
        for step in range(self.compute_halveableness()):
            self.layer_list.append(DepthConvBundle(total_pixel_depth=total_pixel_depth,
                                                   internal_pixel_depth=internal_pixel_depth,
                                                   jump_over_pixels=True))
            self.layer_list.append(InvertiblePolynome())

            self.layer_list.append(SLogGate())
            self.layer_list.append(DepthConvBundle(total_pixel_depth=total_pixel_depth,
                                                   internal_pixel_depth=internal_pixel_depth,
                                                   jump_over_pixels=True))
            self.layer_list.append(InvertiblePolynome())

            total_pixel_depth = total_pixel_depth // 2# + total_pixel_depth % 2
            self.layer_list.append(CutOff(remaining_depth=total_pixel_depth))

            self.layer_list.append(CombineNeighbors())

            internal_pixel_depth = total_pixel_depth
            total_pixel_depth = total_pixel_depth * COMBINE_NEIGHBOR_KERNEL_SIZE_SQ

        print('####Last Pixel Stuff:')
        # last pixel
        # for the last 'pixel' we, make some extra computation
        while total_pixel_depth > LAST_PIXEL_BREAK_DOWN:
            self.layer_list.append(DepthConvBundle(total_pixel_depth=total_pixel_depth,
                                                   internal_pixel_depth=1,
                                                   jump_over_pixels=False))
            self.layer_list.append(InvertiblePolynome())

            self.layer_list.append(SLogGate())
            """
            self.layer_list.append(DepthConvBundle(total_pixel_depth=total_pixel_depth,
                                                   internal_pixel_depth=1,
                                                   jump_over_pixels=False))
            self.layer_list.append(InvertiblePolynome())
            """
            total_pixel_depth = total_pixel_depth // LAST_PIXEL_BREAK_DOWN
            self.layer_list.append(CutOff(remaining_depth=total_pixel_depth))

        print_separator()


    @staticmethod
    def compute_halveableness():
        width = DATA_WIDTH // INITIAL_KERNEL_SIZE
        exp = np.log2(width)
        return int(np.floor(exp))

    def forward(self, x: torch.Tensor):
        pyramid_steps = []
        pyramid_steps_lnorm = []
        waste_steps = []
        waste_steps_lnorm = []

        x = channel_to_last_dim(x)
        x = merge_patches(x, INITIAL_KERNEL_SIZE)

        lnorm_map = torch.zeros_like(x, device=DEVICE)
        #log_norm = torch.zeros(x.shape[0], device=DEVICE)

        for layer in self.layer_list:
            if type(layer) is CutOff:
                x, cut_off = layer(x)
                pyramid_steps.append(cut_off)

                lnorm_map, cut_off_lnorm_map = layer(lnorm_map)
                pyramid_steps_lnorm.append(cut_off_lnorm_map)
            elif type(layer) is CombineNeighbors:
                warn("There need to be as many waste_steps need to align with pyramid_steps")
                x, waste = layer(x)
                waste_steps.append(waste)

                lnorm_map, lnorm_waste_map = layer(lnorm_map)
                waste_steps_lnorm.append(lnorm_waste_map)
            else: # normal procedure
                x, lnorm_map = layer(x, lnorm_map)

            #print(x)
            #print(f'{type(layer)} {x.shape}')
            if False and torch.isnan(x).any():
                print(Batch_Idx)
                #exit(1)

        if self.first_run_done is False:
            self._print_waste(waste_steps)

        self.first_run_done = True
        # top data needs to be appended to the pyramid
        pyramid_steps.append(x)
        pyramid_steps_lnorm.append(lnorm_map)

        return pyramid_steps, pyramid_steps_lnorm

    def _print_waste(self, waste_steps):
        total_waste = 0
        for w in waste_steps:
            if w is None:
                continue
            total_waste += w.shape[1] * w.shape[2]
        print(f'Total Waste {total_waste} of {TOTAL_IMAGE_DIMENSION}  {total_waste/TOTAL_IMAGE_DIMENSION*100:.2f}%')

    def print_parameter(self):
        print_separator()
        for layer in self.layer_list:
            layer.print_parameter()
        print_separator()

    def get_parameter_count(self):
        return sum([layer.get_parameter_count() for layer in self.layer_list])
