import torch
import torch.nn as nn

import math
from se3cnn.SO3 import clebsch_gordan

if torch.cuda.is_available():
    from se3cnn import real_spherical_harmonics


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