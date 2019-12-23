import torch
import torch.nn as nn

from se3cnn.point.radial import CosineBasisModel

from se3cnn.persistent_point.data_hub import DataHub
from se3cnn.persistent_point.periodic_convolution import PeriodicConvolutionWithKernel
from se3cnn.persistent_point.gate import Gate


class EQLayer(nn.Module):
    def __init__(self, data_hub, number_of_the_layer, radial_basis_function_kwargs, gate_kwargs, radial_basis_function=CosineBasisModel, convolution=PeriodicConvolutionWithKernel, gate=Gate, device=torch.device(type='cuda', index=0)):
        super().__init__()
        self.data_hub = data_hub
        self.n = number_of_the_layer
        self.radial_basis_function_kwargs = radial_basis_function_kwargs  # or {'max_radius': 5.0, 'number_of_basis': 100, 'h': 100, 'L': 2, 'act': relu}
        self.gate_kwargs = gate_kwargs  # or {'scalar_activation': relu, 'gate_activation': sigmoid}

        self.radial_basis_trainable_function = radial_basis_function(out_dim=self.data_hub.R_base_offsets[self.n][-1].item(), **radial_basis_function_kwargs).to(device)

        self.convolution = convolution(self.data_hub, self.n)
        if self.data_hub.has_gates[self.n]:
            self.gate = gate(self.data_hub, self.n, **gate_kwargs)

    def forward(self, features):
        rbf_coefficients = self.radial_basis_trainable_function(self.data_hub.radii)
        return self.gate(self.convolution(features, rbf_coefficients)) if hasattr(self, 'gate') else self.convolution(features, rbf_coefficients)


class EQNetwork(nn.Module):
    def __init__(self, representations, radial_basis_functions_kwargs, gate_kwargs, radial_basis_function=CosineBasisModel, convolution=PeriodicConvolutionWithKernel, gate=Gate, has_gates=True, normalization='norm', device=torch.device(type='cuda', index=0)):
        super().__init__()
        number_of_layers = len(representations) - 1

        # region input check
        assert isinstance(has_gates, bool) or \
               (isinstance(has_gates, (list, tuple)) and len(has_gates) == number_of_layers and all(isinstance(has_gate, bool) for has_gate in has_gates)), \
            "has_gates should be specified as a single boolean value or as list/tuple of boolean values that matches number of layers"

        assert isinstance(radial_basis_functions_kwargs, dict) or \
               (isinstance(radial_basis_functions_kwargs, (list, tuple)) and len(radial_basis_functions_kwargs) == number_of_layers and all(isinstance(rbf_args, dict) for rbf_args in radial_basis_functions_kwargs)), \
            "radial_basis_functions_kwargs should be specified as a single dict (shared for all layers) or as list/tuple of dicts - one for each layers"

        assert isinstance(gate_kwargs, dict) or \
               (isinstance(gate_kwargs, (list, tuple)) and len(gate_kwargs) == number_of_layers and all(isinstance(g_args, dict) for g_args in gate_kwargs)), \
            "gate_kwargs should be specified as a single dict (shared for all layers) or as list/tuple of dicts - one for each layers"
        # endregion

        has_gates = [has_gates] * number_of_layers if isinstance(has_gates, bool) else has_gates

        # construct representations, without gates - gates got added in Data Hub where necessary
        # can have mixed specifications (short - multiplicity, long - multiplicity and rotation order) across layers, but within layer it should be consistent
        representations = [[(mul, l) if isinstance(mul, int) else mul for l, mul in enumerate(rs)] for rs in representations]

        self.data_hub = DataHub(representations, has_gates, normalization, device)

        radial_basis_functions_kwargs_list = [radial_basis_functions_kwargs] * number_of_layers if isinstance(radial_basis_functions_kwargs, dict) else radial_basis_functions_kwargs
        gate_kwargs_list = [gate_kwargs] * number_of_layers if isinstance(gate_kwargs, dict) else gate_kwargs

        layers = []
        for i in range(number_of_layers):
            layers.append(EQLayer(self.data_hub, i, radial_basis_functions_kwargs_list[i], gate_kwargs_list[i], radial_basis_function, convolution, gate, device))

        self.layers = nn.Sequential(*layers)

    def forward(self, features, radii_vectors, n_norm, ab_p_to_a, ab_p_to_b):
        self.data_hub(radii_vectors, n_norm, ab_p_to_a, ab_p_to_b)
        return self.layers(features).mean(dim=0)
