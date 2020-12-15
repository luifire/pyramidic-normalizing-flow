import torch

from pyr_flow.model.computation_layers.depth_conv_layer import *
from pyr_flow.model.reshaping.initial_reshaping import merge_patches
from pyr_flow.model.computation_layers.s_log_gate import SLogGate
from pyr_flow.model.computation_layers.tanh_gate import TanhGate
from pyr_flow.model.layer_module import LayerModule
from pyr_flow.model.reshaping.cut_off_layer import CutOff
from pyr_flow.model.computation_layers.conv_bundle import DepthConvBundle
from pyr_flow.model.reshaping.combine_neighboring_info import CombineNeighbors
from pyr_flow.model.computation_layers.invertible_polynomes import InvertiblePolynome
from pyr_flow.model.computation_layers.leaky_relu_gate import LeakyRelu
from pyr_flow.model.computation_layers.bent_identity_gate import BentIdentity
from pyr_flow.model.computation_layers.model_5_post_cut_off_reparam_trick import inverse_reparam_trick

from pyr_flow.utils.functional_utils import channel_to_last_dim

from pyr_flow.misc.misc import *
from pyr_flow.constants import *


class PyramidFlowModel(LayerModule):

    def __init__(self):
        super().__init__()

        self.first_run_done = False

        if MODEL_TYPE == 5:
            layer_list = self._model_type_5()
        else:
            layer_list = self._model_type_not_5()
        self.layer_list = nn.ModuleList(layer_list)

        print_separator()

    def _model_type_5(self):
        layer_list = []

        # we make gray scale at the start
        internal_pixel_depth = 1
        total_pixel_depth = INITIAL_KERNEL_SIZE_SQ  # 9

        """
        9x9x(8x8) -> 3x(8x8) -> 192 -> combine -> non-lin -> 192 -> combine -> divide -> 96
        -> combine / non-lin -> combine divide -> ... -> 48 -> 24 -> 2 -> 6 -> 3 (-> 1)
        """

        #  !!! happens in another way layer_list.append(GrayScale())
        layer_list.append(DepthConvBundle(total_pixel_depth=total_pixel_depth,
                                          internal_pixel_depth=internal_pixel_depth,
                                          jump_over_pixels=False,
                                          conv_type=2))

        total_pixel_depth = 3
        layer_list.append(CutOff(remaining_depth=total_pixel_depth))

        # afterwards we should have 1 pixel of size 192
        for _ in range(3):
            layer_list.append(CombineNeighbors())
            internal_pixel_depth = total_pixel_depth
            total_pixel_depth = total_pixel_depth * COMBINE_NEIGHBOR_KERNEL_SIZE_SQ

        assert internal_pixel_depth * COMBINE_NEIGHBOR_KERNEL_SIZE_SQ == total_pixel_depth
        while total_pixel_depth > 1:
            layer_list.append(DepthConvBundle(total_pixel_depth=total_pixel_depth,
                                              internal_pixel_depth=-1,
                                              jump_over_pixels=False,
                                              conv_type=2))
            total_pixel_depth = total_pixel_depth // 2
            layer_list.append(CutOff(remaining_depth=total_pixel_depth))

        warn('adjust scaling - we start with a 24x24x1 image now!')
        warn('no non-linearity used')
        return layer_list

    def _model_type_not_5(self):
        internal_pixel_depth = PIXEL_DEPTH
        total_pixel_depth = INITIAL_KERNEL_SIZE_SQ * internal_pixel_depth

        layer_list = []

        """
                layer_list.append(DepthConvBundle("1", channel_count=channel_size, bundle_size=bundle_size,
                                                       jump_over_pixels=True))

                layer_list.append(InvertiblePolynome())

                layer_list.append(SLogGate())
                layer_list.append(DepthConvBundle("2", channel_count=channel_size, bundle_size=bundle_size_2,
                                                       jump_over_pixels=True))
                layer_list.append(InvertiblePolynome())
                """
        print_separator()
        print('Creating Pyramid Flow Network')
        for step in range(self.compute_halveableness()):

            # bau einen eye layer, wo man die diagonale lernen kann. und dann? David?

            layer_list.append(DepthConvBundle(total_pixel_depth=total_pixel_depth,
                                              internal_pixel_depth=internal_pixel_depth,
                                              jump_over_pixels=True))

            if MODEL_TYPE == 1:
                layer_list.append(SLogGate())
            elif MODEL_TYPE == 2:
                layer_list.append(TanhGate())
                layer_list.append(InvertiblePolynome())
            elif MODEL_TYPE == 3:
                layer_list.append(LeakyRelu())
            elif MODEL_TYPE == 4:
                layer_list.append(BentIdentity())

            total_pixel_depth = total_pixel_depth // 2
            layer_list.append(CutOff(remaining_depth=total_pixel_depth))

            layer_list.append(CombineNeighbors())

            internal_pixel_depth = total_pixel_depth
            total_pixel_depth = total_pixel_depth * COMBINE_NEIGHBOR_KERNEL_SIZE_SQ

        print('####Last Pixel Stuff:')
        # last pixel
        # for the last 'pixel' we, make some extra computation
        while total_pixel_depth > LAST_PIXEL_BREAK_DOWN:
            pixel_jumper = 1
            if total_pixel_depth > 50:
                pixel_jumper = total_pixel_depth // 2

            layer_list.append(DepthConvBundle(total_pixel_depth=total_pixel_depth,
                                              internal_pixel_depth=pixel_jumper,
                                              jump_over_pixels=True))

            if MODEL_TYPE == 1:
                layer_list.append(SLogGate())
            elif MODEL_TYPE == 2:
                layer_list.append(TanhGate())
                layer_list.append(InvertiblePolynome())
            elif MODEL_TYPE == 3:
                layer_list.append(LeakyRelu())
            elif MODEL_TYPE == 4:
                layer_list.append(BentIdentity())

            total_pixel_depth = total_pixel_depth // LAST_PIXEL_BREAK_DOWN
            layer_list.append(CutOff(remaining_depth=total_pixel_depth))

        # remove last tanh gate
        if MODEL_TYPE in [2, 3, 4]:
            layer_list.remove(layer_list[-1])
            layer_list.remove(layer_list[-1])
        return layer_list

    @staticmethod
    def compute_halveableness():
        width = DATA_WIDTH // INITIAL_KERNEL_SIZE
        exp = np.log2(width)
        return int(np.floor(exp))

    def _forward_model_5(self, x):
        pyramid_steps = []
        pyramid_steps_lnorm = []

        # make Gray Scale
        x = channel_to_last_dim(x)
        x = (x[:, :, :, 0] + x[:, :, :, 1] + x[:, :, :, 2]) / 3
        x = x.unsqueeze(3)
        x = merge_patches(x, INITIAL_KERNEL_SIZE)

        lnorm_map = torch.zeros_like(x, device=DEVICE)

        for layer in self.layer_list:
            if type(layer) is CutOff:
                x, cut_off = layer(x)

                lnorm_map, cut_off_lnorm_map = layer(lnorm_map)

                #  !!!!!!!!! after cut of, we perform inverse param trick!!!!
                cut_off, cut_off_lnorm_map = inverse_reparam_trick(cut_off, cut_off_lnorm_map)

                pyramid_steps.append(cut_off)
                pyramid_steps_lnorm.append(cut_off_lnorm_map)
            elif type(layer) is CombineNeighbors:
                x, _ = layer(x)
                lnorm_map, _ = layer(lnorm_map)

            else:  # normal procedure
                x, lnorm_map = layer(x, lnorm_map)

        self.first_run_done = True

        # top data needs to be appended to the pyramid
        #  !!!!!!!!! for last values, we also perform inverse param trick!!!!
        cut_off, cut_off_lnorm_map = inverse_reparam_trick(x, lnorm_map)

        pyramid_steps.append(x)
        pyramid_steps_lnorm.append(lnorm_map)

        return pyramid_steps, pyramid_steps_lnorm

    def _forward_not_model_5(self, x: torch.Tensor):
        pyramid_steps = []
        pyramid_steps_lnorm = []
        waste_steps = []
        waste_steps_lnorm = []

        x = channel_to_last_dim(x)
        x = merge_patches(x, INITIAL_KERNEL_SIZE)

        lnorm_map = torch.zeros_like(x, device=DEVICE)
        # log_norm = torch.zeros(x.shape[0], device=DEVICE)

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
            else:  # normal procedure
                x, lnorm_map = layer(x, lnorm_map)

            # print(x)
            # print(f'{type(layer)} {x.shape}')
            if False and torch.isnan(x).any():
                print(Batch_Idx)
                # exit(1)

        if self.first_run_done is False:
            self._print_waste(waste_steps)

        self.first_run_done = True
        # top data needs to be appended to the pyramid
        pyramid_steps.append(x)
        pyramid_steps_lnorm.append(lnorm_map)

        return pyramid_steps, pyramid_steps_lnorm

    def forward(self, x):
        if MODEL_TYPE == 5:
            return self._forward_model_5(x)
        else:
            return self._forward_not_model_5(x)

    def _print_waste(self, waste_steps):
        total_waste = 0
        for w in waste_steps:
            if w is None:
                continue
            total_waste += w.shape[1] * w.shape[2]
        print(f'Total Waste {total_waste} of {TOTAL_IMAGE_DIMENSION}  {total_waste / TOTAL_IMAGE_DIMENSION * 100:.2f}%')

    def print_parameter(self):
        print_separator()
        with torch.no_grad():
            for layer in self.layer_list:
                layer.print_parameter()
        print_separator()

    def get_parameter_count(self):
        return sum([layer.get_parameter_count() for layer in self.layer_list])
