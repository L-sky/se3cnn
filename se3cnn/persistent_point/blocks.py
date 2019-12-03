import torch
import torch.nn as nn

import math

import se3cnn.SO3 as SO3
from se3cnn.non_linearities.rescaled_act import relu, sigmoid

from se3cnn.SO3 import clebsch_gordan


class DataHub(nn.Module):
    def __init__(self, representations, normalization, device):
        super().__init__()
        assert len(self.representations) > 1, "there should be at list two representations to form a network"
        self.representations = representations  # layer representations

        self.max_l_out = None                   # Max l_out in a network -> for Clebsch Gordan coefficients
        self.max_l_in = None                    # Max l_in in a network -> for Clebsch Gordan coefficients
        self.max_l = None                       # Max l in a network -> for Spherical Harmonics
        self.cg = None                          # Clebsch-Gordan coefficients that cover all layers of a network, without gaps

        self.l_out_list = []
        self.l_in_list = []
        self.mul_out_list = []
        self.mul_in_list = []
        self.norm_coef_list = []

        # jump points for C++/CUDA
        self.output_base_offsets = None
        self.grad_base_offsets = None
        self.cg_base_offsets = None
        self.features_base_offsets = None

        # data specific to particular input
        self.Ys = None                          # Spherical harmonics that cover all layers of a network
        self.radii = None                       # Absolute distances
        self.map_ab_p_to_a = None
        self.map_ab_p_to_b = None

        assert isinstance(normalization, str), "pass normalization type as a string, either 'norm' or 'component'"
        assert normalization in ['norm', 'component'], "normalization needs to be either 'norm' or 'component'"
        self.normalization_type = normalization

        # As it is, it will only support gpu, but it may still be useful to select which one of them to use
        self.device = device

        # Setup values
        max_l_out = 0
        max_l_in = 0
        max_l = 0

        for Rs_in, Rs_out in zip(self.representations[:-1], self.representations[1:]):
            # running values are need to make sure max_l is not higher that needed (consequent layers)
            # e.g, rotational orders: [4, 6, 4] -> 12 (without), but [4, 6, 4] -> 10 (with)
            running_max_l_out = 0
            running_max_l_in = 0

            tmp_l_out_list = []
            tmp_l_in_list = []
            tmp_mul_out_list = []
            tmp_mul_in_list = []

            norm_coef = torch.zeros((len(Rs_out), len(Rs_in), 2), device=self.device)

            for mul_out, l_out, _ in Rs_out:
                tmp_mul_out_list.append(mul_out)
                tmp_l_out_list.append(l_out)
                running_max_l_out = max(l_out, running_max_l_out)

            for mul_in, l_in, _ in Rs_in:
                tmp_mul_in_list.append(mul_in)
                tmp_l_in_list.append(l_in)
                running_max_l_in = max(l_in, running_max_l_in)

            max_l_out = max(running_max_l_out, max_l_out)
            max_l_in = max(running_max_l_in, max_l_in)
            max_l = max(running_max_l_out + running_max_l_in, max_l)

            for i, (_, l_out, _) in enumerate(Rs_out):
                num_summed_elements = 0
                for mul_in, l_in, _ in enumerate(Rs_in):
                    # TODO: account for parity, when it would be supported in CUDA code
                    num_summed_elements += mul_in * (2*min(l_out, l_in) + 1)
                for j, (mul_in, l_in, _) in enumerate(Rs_in):
                    norm_coef[i, j, 0] = self.lm_normalization(l_out, l_in) / math.sqrt(mul_in)
                    norm_coef[i, j, 1] = self.lm_normalization(l_out, l_in) / math.sqrt(num_summed_elements)

            # so annoying that there is no uint32 data type in PyTorch
            self.l_out_list.append(torch.tensor(tmp_l_out_list, dtype=torch.int32, device=self.device))
            self.l_in_list.append(torch.tensor(tmp_l_in_list, dtype=torch.int32, device=self.device))
            self.mul_out_list.append(torch.tensor(tmp_mul_out_list, dtype=torch.int32, device=self.device))
            self.mul_in_list.append(torch.tensor(tmp_mul_in_list, dtype=torch.int32, device=self.device))

            # TODO: or move it, and calculate from lists ?
            self.norm_coef_list.append(norm_coef)

            # TODO: still need to calculate cg and offsets

    def lm_normalization(self, l_out, l_in):
        # put 2l_in+1 to keep the norm of the m vector constant
        # put 2l_ou+1 to keep the variance of each m component constant
        # sum_m Y_m^2 = (2l+1)/(4pi)  and  norm(Q) = 1  implies that norm(QY) = sqrt(1/4pi)
        lm_norm = None
        if self.normalization_type == 'norm':
            lm_norm = math.sqrt(2 * l_in + 1) * math.sqrt(4 * math.pi)
        elif self.normalization_type == 'component':
            lm_norm = math.sqrt(2 * l_out + 1) * math.sqrt(4 * math.pi)
        return lm_norm


    def forward(self, radii_vectors):
        # self.radii = torch.nn.functional.normalize(radii_vectors, p=2, dim=-1) # need something else
        self.Ys = SO3.spherical_harmonics_xyz(self.max_l, radii_vectors)  # TODO: wrong as is now, correct radii_vectors


        max_l = max(order)
        out = xyz.new_empty(((max_l + 1)*(max_l + 1), xyz.size(0)))                                    # [filters, batch_size]
        xyz_unit = torch.nn.functional.normalize(xyz, p=2, dim=-1)
        real_spherical_harmonics.rsh(out, xyz_unit)
        norm_coef = [elem for lh in range((max_l+1)//2) for elem in [1.]*(4*lh + 1) + [-1.]*(4*lh+3)]  # (-1)^L same as (pi-theta) -> (-1)^(L+m) and 'quantum' norm (-1)^m combined  # h - halved
        if max_l % 2 == 0:
            norm_coef.extend([1.]*(2*max_l + 1))
        norm_coef = torch.tensor(norm_coef, device=device).unsqueeze(1)
        out.mul_(norm_coef)
        if order != list(range(max_l+1)):
            keep_rows = torch.zeros(out.size(0), dtype=torch.bool)
            [keep_rows[(l*l):((l+1)*(l+1))].fill_(True) for l in order]
            out = out[keep_rows.to(device)]




class EQNetwork(torch.nn.Module):
    def __init__(self, representations, max_radius):
        super().__init__()

        representations = [[(mul, l) for l, mul in enumerate(rs)] for rs in representations]

        R = partial(CosineBasisModel, max_radius=max_radius, number_of_basis=10, h=100, L=2, act=relu)
        K = partial(Kernel, RadialModel=R)
        C = partial(PeriodicConvolutionPrep, Kernel=K)

        self.firstlayers = torch.nn.ModuleList([
            GatedBlock(Rs_in, Rs_out, relu, sigmoid, C)
            for Rs_in, Rs_out in zip(representations, representations[1:])
        ])

    def forward(self, radii, bs_slice, charges):
        p = next(self.parameters())
        # features = p.new_ones(len(charges), 1)
        features = p.new_tensor(charges.numpy()/94 - 0.5).unsqueeze(1)

        for i, m in enumerate(self.firstlayers):
            features = m(features.div(2), radii, bs_slice)

        features = torch.mean(features, dim=0)
        return features