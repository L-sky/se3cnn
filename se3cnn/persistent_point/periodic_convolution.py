import torch
import torch.nn as nn

if torch.cuda.is_available():
    from se3cnn import pconv_with_kernel


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

