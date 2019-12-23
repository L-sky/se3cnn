import torch
import torch.nn as nn

# import numpy as np

import math

from se3cnn.SO3 import clebsch_gordan
from se3cnn.point.radial import CosineBasisModel

if torch.cuda.is_available():
    from se3cnn import real_spherical_harmonics
    from se3cnn import pconv_with_kernel


class DataHub(nn.Module):
    def __init__(self, representations, has_gates, normalization='norm', device=torch.device(type='cuda', index=0)):
        super().__init__()
        assert len(representations) > 1, "there should be at list two representations to form a network"
        self.representations = representations  # layer representations

        assert isinstance(normalization, str), "pass normalization type as a string, either 'norm' or 'component'"
        assert normalization in ['norm', 'component'], "normalization needs to be either 'norm' or 'component'"
        self.normalization_type = normalization

        # As it is, it will only support gpu, but it may still be useful to select which one of them to use
        self.device = device

        # presence of the gate affects number of filters
        self.has_gates = has_gates
        self.gates_count_list = []

        # Network properties - calculate once (init)
        self.max_l_out = None                                       # Max l_out in a network -> for Clebsch Gordan coefficients
        self.max_l_in = None                                        # Max l_in in a network -> for Clebsch Gordan coefficients
        self.max_l = None                                           # Max l in a network -> for Spherical Harmonics

        self.cg = None                                              # Clebsch-Gordan coefficients

        self.l_out_list = []
        self.l_in_list = []
        self.mul_out_list = []
        self.mul_in_list = []

        self.norm_coef_list = []

        self.rsh_norm_coefficients = None                           # 'sign' normalization for Spherical harmonics

        # jump points for C++/CUDA - calculate once (init)
        self.R_base_offsets = []
        self.grad_base_offsets = []
        self.cg_base_offsets = []
        self.features_base_offsets = []

        # tied to input - recalculate on each forward call
        self.Ys = None                                              # Spherical harmonics
        self.radii = None                                           # Absolute distances
        self.n_norm = None
        self.map_ab_p_to_a = None
        self.map_ab_p_to_b = None

        self.find_max_l_out_in()                                    # max_l_out, max_l_in, max_l
        self.calculate_l_out_in_and_mul_out_in_lists()              # l_out_list, l_in_list, mul_out_list, mul_in_list
        self.calculate_normalization_coefficients()                 # norm_coef_list
        self.calculate_rsh_normalization_coefficients()             # rsh_norm_coefficients # TODO: move rsh normalization to CUDA code (template parameter?)
        self.calculate_clebsch_gordan_coefficients_and_offsets()    # cg, features_base_offsets
        self.calculate_offsets()                                    # output_base_offsets, grad_base_offsets, features_base_offsets

    def find_max_l_out_in(self):
        max_l_out = max_l_in = max_l = 0

        for Rs_in, Rs_out in zip(self.representations[:-1], self.representations[1:]):
            # running values are need to make sure max_l is not higher that needed (consequent layers)
            # e.g, rotational orders: [4, 6, 4] -> 12 (without), but [4, 6, 4] -> 10 (with)
            running_max_l_out = max([l_out for _, l_out in Rs_out])
            running_max_l_in = max([l_in for _, l_in in Rs_in])

            max_l_out = max(running_max_l_out, max_l_out)
            max_l_in = max(running_max_l_in, max_l_in)
            max_l = max(running_max_l_out + running_max_l_in, max_l)

        assert max_l <= 10, "l > 10 is not supported yet"
        self.max_l_out = max_l_out
        self.max_l_in = max_l_in
        self.max_l = max_l

    def calculate_l_out_in_and_mul_out_in_lists(self):
        for Rs_in, Rs_out, has_gate in zip(self.representations[:-1], self.representations[1:], self.has_gates):
            tmp_l_out_list = [l_out for _, l_out in Rs_out]
            tmp_l_in_list = [l_in for _, l_in in Rs_in]
            tmp_mul_out_list = [mul_out for mul_out, _ in Rs_out]
            tmp_mul_in_list = [mul_in for mul_in, _ in Rs_in]

            if has_gate:
                gates_count = 0
                if tmp_l_out_list[0] != 0:                                                  # expected sorted representations and consequently lists
                    tmp_l_out_list = [0] + tmp_l_out_list
                    gates_count = sum(tmp_mul_out_list)
                    tmp_mul_out_list = tmp_mul_out_list + [gates_count]
                elif len(tmp_l_out_list) > 1:                                               # not only scalars, so non-zero number of gates
                    gates_count = sum(tmp_mul_out_list[1:])
                    tmp_mul_out_list[0] += sum(tmp_mul_out_list[1:])

                self.gates_count_list.append(gates_count)
            else:
                self.gates_count_list.append(0)

            # so annoying that there is no uint32 data type in PyTorch
            self.l_out_list.append(torch.tensor(tmp_l_out_list, dtype=torch.int32, device=self.device))
            self.l_in_list.append(torch.tensor(tmp_l_in_list, dtype=torch.int32, device=self.device))
            self.mul_out_list.append(torch.tensor(tmp_mul_out_list, dtype=torch.int32, device=self.device))
            self.mul_in_list.append(torch.tensor(tmp_mul_in_list, dtype=torch.int32, device=self.device))

    def calculate_normalization_coefficients(self):
        def lm_normalization(l_out, l_in):
            # put 2l_in+1 to keep the norm of the m vector constant
            # put 2l_ou+1 to keep the variance of each m component constant
            # sum_m Y_m^2 = (2l+1)/(4pi)  and  norm(Q) = 1  implies that norm(QY) = sqrt(1/4pi)
            lm_norm = None
            if self.normalization_type == 'norm':
                lm_norm = math.sqrt(2 * l_in + 1) * math.sqrt(4 * math.pi)
            elif self.normalization_type == 'component':
                lm_norm = math.sqrt(2 * l_out + 1) * math.sqrt(4 * math.pi)
            return lm_norm

        for l_in_layer, mul_in_layer, l_out_layer, mul_out_layer in zip(self.l_in_list, self.mul_in_list, self.l_out_list, self.mul_out_list):
            norm_coef = torch.zeros((len(l_out_layer), len(l_in_layer), 2), device=self.device)
            for i, l_out in enumerate(l_out_layer):
                num_summed_elements = sum([mul_in * (2 * min(l_out, l_in) + 1) for mul_in, l_in in zip(mul_in_layer, l_in_layer)])  # (l_out + l_in) - |l_out - l_in| = 2*min(l_out, l_in)
                for j, (mul_in, l_in) in enumerate(zip(mul_in_layer, l_in_layer)):
                    norm_coef[i, j, 0] = lm_normalization(l_out, l_in) / math.sqrt(mul_in)
                    norm_coef[i, j, 1] = lm_normalization(l_out, l_in) / math.sqrt(num_summed_elements)
            self.norm_coef_list.append(norm_coef)

    def calculate_rsh_normalization_coefficients(self):
        # (-1)^L same as (pi-theta) -> (-1)^(L+m) and 'quantum' norm (-1)^m combined  # h - halved
        rsh_norm_coef = [elem for lh in range((self.max_l + 1) // 2) for elem in [1.] * (4 * lh + 1) + [-1.] * (4 * lh + 3)]
        if self.max_l % 2 == 0:
            rsh_norm_coef.extend([1.] * (2 * self.max_l + 1))
        self.rsh_norm_coefficients = torch.tensor(rsh_norm_coef, device=self.device).unsqueeze(1)

    def calculate_clebsch_gordan_coefficients_and_offsets(self):
        tmp_cg_base_offsets_list = []
        tmp_cg_position = 0
        for l_out in range(self.max_l_out + 1):
            for l_in in range(self.max_l_in + 1):
                tmp_cg_base_offsets_list.append(tmp_cg_position)
                for l_f in range(abs(l_out - l_in), l_out + l_in + 1):
                    tmp_cg_position += (2 * l_out + 1) * (2 * l_in + 1) * (2 * l_f + 1)

        self.cg = torch.zeros(tmp_cg_position, device=self.device)      # last tmp_cg_pos (not stored in offsets) = size

        for l_out in range(self.max_l_out + 1):
            for l_in in range(self.max_l_in + 1):
                tmp_cg_base_pos = tmp_cg_base_offsets_list[l_out * (self.max_l_in + 1) + l_in]
                for l_f in range(abs(l_out - l_in), l_out + l_in + 1):
                    tmp_cg_flat = clebsch_gordan(l_out, l_in, l_f).view(-1).type(torch.get_default_dtype()).to(self.device)
                    self.cg[tmp_cg_base_pos:tmp_cg_base_pos + tmp_cg_flat.shape[0]] = tmp_cg_flat
                    tmp_cg_base_pos += tmp_cg_flat.shape[0]

        self.cg_base_offsets = torch.tensor(tmp_cg_base_offsets_list, dtype=torch.int32, device=self.device)

    def calculate_offsets(self):
        from itertools import accumulate
        for l_in_layer, mul_in_layer, l_out_layer, mul_out_layer in zip(self.l_in_list, self.mul_in_list, self.l_out_list, self.mul_out_list):
            tmp_R_base_offset = list(accumulate([mul_out * mul_in * (2 * min(l_out, l_in) + 1) for (mul_out, l_out) in zip(mul_out_layer, l_out_layer) for (mul_in, l_in) in zip(mul_in_layer, l_in_layer)]))
            tmp_grad_base_offset = list(accumulate(mul_out * (2 * l_out + 1) for (mul_out, l_out) in zip(mul_out_layer, l_out_layer)))
            tmp_features_base_offset = list(accumulate(mul_in * (2 * l_in + 1) for (mul_in, l_in) in zip(mul_in_layer, l_in_layer)))

            tmp_R_base_offset.insert(0, 0)
            tmp_grad_base_offset.insert(0, 0)
            tmp_features_base_offset.insert(0, 0)

            self.R_base_offsets.append(torch.tensor(tmp_R_base_offset, dtype=torch.int32, device=self.device))
            self.grad_base_offsets.append(torch.tensor(tmp_grad_base_offset, dtype=torch.int32, device=self.device))
            self.features_base_offsets.append(torch.tensor(tmp_features_base_offset, dtype=torch.int32, device=self.device))

    def forward(self, radii_vectors, n_norm, ab_p_to_a, ab_p_to_b):
        # calculate absolute distances
        self.radii = radii_vectors.norm(p=2, dim=-1)

        # calculate Spherical harmonics
        self.Ys = radii_vectors.new_empty(((self.max_l + 1) * (self.max_l + 1), radii_vectors.size(0)))  # [filters, batch_size]
        real_spherical_harmonics.rsh(self.Ys, radii_vectors / self.radii.clamp_min(1e-12).unsqueeze(1).expand_as(radii_vectors))
        self.Ys.mul_(self.rsh_norm_coefficients)

        # assign input size based normalization coefficients
        self.n_norm = n_norm

        # assign inverse maps
        self.map_ab_p_to_a = ab_p_to_a
        self.map_ab_p_to_b = ab_p_to_b


class PeriodicConvolutionWithKernel(nn.Module):
    def __init__(self, data_hub, number_of_the_layer):
        super().__init__()
        self.data_hub = data_hub
        self.n = number_of_the_layer

    def forward(self, features, radial_basis_function_coefficients):
        return PeriodicConvolutionWithKernelFunction.apply(features, radial_basis_function_coefficients, self.data_hub, self.n)


class PeriodicConvolutionWithKernelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, radial_basis_function_coefficients, data_hub, n):
        # TODO: trace deeper the deal with non-deterministic error (on backward) without .clone() when number of points is 1
        if features.shape[0] == 1:
            F = features.transpose(0, 1).contiguous().clone()                                   # [b*, (l_in, v, j)] -> [(l_in, v, j), b*]
            R = radial_basis_function_coefficients.transpose(0, 1).contiguous().clone()         # [(a, b'), (l_out, l_in, f(l), u, v)] -> [(l_out, l_in, f(l), u, v), (a, b')]
        else:
            F = features.transpose(0, 1).contiguous()
            R = radial_basis_function_coefficients.transpose(0, 1).contiguous()

        B = pconv_with_kernel.forward_stage_one(
            data_hub.norm_coef_list[n],                 # W
            data_hub.cg,                                # C
            F,                                          # F
            data_hub.Ys,                                # Y
            R,                                          # R
            data_hub.radii,                             # radii
            data_hub.l_out_list[n],                     # L_out_list
            data_hub.l_in_list[n],                      # L_in_list
            data_hub.mul_out_list[n],                   # u_sizes
            data_hub.mul_in_list[n],                    # v_sizes
            data_hub.grad_base_offsets[n],              # output_base_offsets
            data_hub.cg_base_offsets,                   # C_offsets
            data_hub.features_base_offsets[n],          # F_base_offsets
            data_hub.R_base_offsets[n],                 # R_base_offsets
            data_hub.map_ab_p_to_b,                     # ab_p_to_b
            data_hub.max_l_in + 1)                      # l_in_max_net_bound

        B = B.transpose_(0, 1).contiguous()                                                 # [(l_out, u, i), (a, b')] -> [(a, b'), (l_out, u, i)]
        F_next = B.new_zeros((F.shape[1], B.shape[1]))                                      # [a, (l_out, u, i)]
        F_next.index_add_(dim=0, index=data_hub.map_ab_p_to_a.type(torch.long), source=B)   # [(a, b'), (l_out, u, i)] -> [a, (l_out, u, i)]
        F_next.mul_(data_hub.n_norm.unsqueeze(1).expand_as(F_next))
        del B

        if features.requires_grad or radial_basis_function_coefficients.requires_grad:
            ctx.save_for_backward(F, R)                 # transposed
            ctx.data_hub = data_hub
            ctx.n = n

        return F_next

    @staticmethod
    def backward(ctx, grad_output):
        F, R = ctx.saved_tensors  # transposed

        data_hub = ctx.data_hub
        n = ctx.n

        # TODO: trace deeper the deal with non-deterministic error without .clone() when number of points is 1
        if F.shape[1] == 1:
            G = grad_output.transpose(0, 1).contiguous().clone()
        else:
            G = grad_output.transpose(0, 1).contiguous()

        G.mul_(data_hub.n_norm.unsqueeze(0).expand_as(G))

        B_grad = pconv_with_kernel.backward_F_stage_one(
            data_hub.norm_coef_list[n],                 # W
            data_hub.cg,                                # C
            G,                                          # G
            data_hub.Ys,                                # Y
            R,                                          # R
            data_hub.radii,                             # radii
            data_hub.l_out_list[n],                     # L_out_list
            data_hub.l_in_list[n],                      # L_in_list
            data_hub.mul_out_list[n],                   # u_sizes
            data_hub.mul_in_list[n],                    # v_sizes
            data_hub.features_base_offsets[n],          # output_base_offsets
            data_hub.cg_base_offsets,                   # C_offsets
            data_hub.grad_base_offsets[n],              # G_base_offsets
            data_hub.R_base_offsets[n],                 # R_base_offsets
            data_hub.map_ab_p_to_a,                     # ab_p_to_a
            data_hub.max_l_in + 1)                      # l_in_max_net_bound

        B_grad = B_grad.transpose_(0, 1).contiguous()                                               # [(a, b'), (l_in, v. j)]
        F_grad = B_grad.new_zeros((G.shape[1], B_grad.shape[1]))                                    # [b*, (l_in, v. j)]
        F_grad.index_add_(dim=0, index=data_hub.map_ab_p_to_b.type(torch.long), source=B_grad)      # [(a, b'), (l_in, v. j)] -> [b*, (l_in, v. j)]
        del B_grad

        R_grad = pconv_with_kernel.backward_R(
            data_hub.norm_coef_list[n],                 # W
            data_hub.cg,                                # C
            G,                                          # G
            F,                                          # F
            data_hub.Ys,                                # Y
            data_hub.radii,                             # radii
            data_hub.l_out_list[n],                     # L_out_list
            data_hub.l_in_list[n],                      # L_in_list
            data_hub.mul_out_list[n],                   # u_sizes
            data_hub.mul_in_list[n],                    # v_sizes
            data_hub.R_base_offsets[n],                 # output_base_offsets
            data_hub.cg_base_offsets,                   # C_offsets
            data_hub.grad_base_offsets[n],              # G_base_offsets
            data_hub.features_base_offsets[n],          # F_base_offsets
            data_hub.map_ab_p_to_a,                     # ab_p_to_a
            data_hub.map_ab_p_to_b,                     # ab_p_to_b
            data_hub.max_l_in + 1)                      # l_in_max_net_bound

        R_grad = R_grad.transpose_(0, 1).contiguous()

        return F_grad, R_grad, None, None


class Gate(nn.Module):
    def __init__(self, data_hub, number_of_the_layer, scalar_activation, gate_activation):
        super().__init__()
        self.data_hub = data_hub
        self.n = number_of_the_layer
        self.gates_count = self.data_hub.gates_count_list[self.n]
        self.scalar_activation = scalar_activation
        self.gate_activation = gate_activation

    def forward(self, features):
        # TODO: move this to CUDA (blocks over l_out)
        out = features.new_empty((features.shape[0], features.shape[1] - self.gates_count))
        filters_l_zero_count = self.data_hub.grad_base_offsets[self.n][1] - self.gates_count
        out[:, :filters_l_zero_count] = self.scalar_activation(features[:, self.gates_count:self.data_hub.grad_base_offsets[self.n][1]])        # scalars, l = 0
        if self.gates_count > 0:
            l_out_list = self.data_hub.l_out_list[self.n]
            mul_out_list = self.data_hub.mul_out_list[self.n]
            features_offsets = self.data_hub.grad_base_offsets[self.n]                                                                          # intentional

            gates = self.gate_activation(features[:, :self.gates_count])
            gates_offset = 0
            out_offset = filters_l_zero_count
            for l_out, u_size, f_end, f_start in zip(l_out_list[1:], mul_out_list[1:], features_offsets[2:], features_offsets[1:-1]):
                i_size = 2*l_out + 1
                out[:, out_offset:out_offset+u_size*i_size] = features[:, f_start:f_end] * gates[:, gates_offset:gates_offset+u_size].unsqueeze(2).expand(-1, -1, i_size).reshape(-1, u_size*i_size)
                gates_offset += u_size
                out_offset += u_size * i_size

        return out


class EQLayer(nn.Module):
    def __init__(self, data_hub, number_of_the_layer, radial_basis_function_kwargs, gate_kwargs, radial_basis_function=CosineBasisModel, convolution=PeriodicConvolutionWithKernel, gate=Gate, device=torch.device(type='cuda', index=0)):
        super().__init__()
        self.data_hub = data_hub
        self.n = number_of_the_layer
        self.radial_basis_function_kwargs = radial_basis_function_kwargs # or {'max_radius': 5.0, 'number_of_basis': 100, 'h': 100, 'L': 2, 'act': relu}
        self.gate_kwargs = gate_kwargs # or {'scalar_activation': relu, 'gate_activation': sigmoid}

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
