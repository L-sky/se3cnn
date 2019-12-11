#include <torch/extension.h>

void forward_stage_one_cuda(
        torch::Tensor output,
        torch::Tensor W,
        torch::Tensor C,
        torch::Tensor F,
        torch::Tensor Y,
        torch::Tensor R,
        torch::Tensor radii,
        torch::Tensor L_out_list,
        torch::Tensor L_in_list,
        torch::Tensor u_sizes,
        torch::Tensor v_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor C_offsets,
        torch::Tensor F_base_offsets,
        torch::Tensor R_base_offsets,
        torch::Tensor ab_p_to_b,
        const uint32_t l_in_max_net_bound);

void backward_F_stage_one_cuda(
        torch::Tensor output,
        torch::Tensor W,
        torch::Tensor C,
        torch::Tensor G,
        torch::Tensor Y,
        torch::Tensor R,
        torch::Tensor radii,
        torch::Tensor L_out_list,
        torch::Tensor L_in_list,
        torch::Tensor u_sizes,
        torch::Tensor v_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor C_offsets,
        torch::Tensor G_base_offsets,
        torch::Tensor R_base_offsets,
        torch::Tensor ab_p_to_a,
        const uint32_t l_in_max_net_bound);

void backward_R_cuda(
        torch::Tensor output,
        torch::Tensor W,
        torch::Tensor C,
        torch::Tensor G,
        torch::Tensor F,
        torch::Tensor Y,
        torch::Tensor radii,
        torch::Tensor L_out_list,
        torch::Tensor L_in_list,
        torch::Tensor u_sizes,
        torch::Tensor v_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor C_offsets,
        torch::Tensor G_base_offsets,
        torch::Tensor F_base_offsets,
        torch::Tensor ab_p_to_a,
        torch::Tensor ab_p_to_b,
        const uint32_t l_in_max_net_bound);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT_DTYPE(x) AT_ASSERTM(x.dtype() == torch::kFloat64 || x.dtype() == torch::kFloat32, #x " must be either float32 or float64")
#define CHECK_INT_DTYPE(x) AT_ASSERTM(x.dtype() == torch::kInt32, #x " must be int32")

#define CHECK_FLOAT_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT_DTYPE(x);
#define CHECK_INT_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT_DTYPE(x);


torch::Tensor forward_stage_one(
        torch::Tensor W,
        torch::Tensor C,
        torch::Tensor F,
        torch::Tensor Y,
        torch::Tensor R,
        torch::Tensor radii,
        torch::Tensor L_out_list,
        torch::Tensor L_in_list,
        torch::Tensor u_sizes,
        torch::Tensor v_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor C_offsets,
        torch::Tensor F_base_offsets,
        torch::Tensor R_base_offsets,
        torch::Tensor ab_p_to_b,
        const uint32_t l_in_max_net_bound
){
    CHECK_FLOAT_INPUT(W);
    CHECK_FLOAT_INPUT(C);
    CHECK_FLOAT_INPUT(F);
    CHECK_FLOAT_INPUT(Y);
    CHECK_FLOAT_INPUT(R);
    CHECK_FLOAT_INPUT(radii);
    CHECK_INT_INPUT(L_out_list);
    CHECK_INT_INPUT(L_in_list);
    CHECK_INT_INPUT(u_sizes);
    CHECK_INT_INPUT(v_sizes);
    CHECK_INT_INPUT(output_base_offsets);
    CHECK_INT_INPUT(C_offsets);
    CHECK_INT_INPUT(F_base_offsets);
    CHECK_INT_INPUT(R_base_offsets);
    CHECK_INT_INPUT(ab_p_to_b);

    const uint32_t lout_ui_size = (uint32_t) output_base_offsets[output_base_offsets.size(0)-1].item<int32_t>();
    const uint32_t ab_p_size = (uint32_t) ab_p_to_b.size(0);

    torch::Tensor output = torch::zeros({lout_ui_size, ab_p_size}, W.options()); // |(l_out, u, i), (a, b_p)|

    forward_stage_one_cuda(output, W, C, F, Y, R, radii,
                           L_out_list, L_in_list, u_sizes, v_sizes,
                           output_base_offsets, C_offsets, F_base_offsets, R_base_offsets,
                           ab_p_to_b,
                           (uint32_t) l_in_max_net_bound);
    return output;
}


torch::Tensor backward_F_stage_one(
        torch::Tensor W,
        torch::Tensor C,
        torch::Tensor G,
        torch::Tensor Y,
        torch::Tensor R,
        torch::Tensor radii,
        torch::Tensor L_out_list,
        torch::Tensor L_in_list,
        torch::Tensor u_sizes,
        torch::Tensor v_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor C_offsets,
        torch::Tensor G_base_offsets,
        torch::Tensor R_base_offsets,
        torch::Tensor ab_p_to_a,
        const int32_t l_in_max_net_bound
){
    CHECK_FLOAT_INPUT(W);
    CHECK_FLOAT_INPUT(C);
    CHECK_FLOAT_INPUT(G);
    CHECK_FLOAT_INPUT(Y);
    CHECK_FLOAT_INPUT(R);
    CHECK_FLOAT_INPUT(radii);
    CHECK_INT_INPUT(L_out_list);
    CHECK_INT_INPUT(L_in_list);
    CHECK_INT_INPUT(u_sizes);
    CHECK_INT_INPUT(v_sizes);
    CHECK_INT_INPUT(output_base_offsets);
    CHECK_INT_INPUT(C_offsets);
    CHECK_INT_INPUT(G_base_offsets);
    CHECK_INT_INPUT(R_base_offsets);
    CHECK_INT_INPUT(ab_p_to_a);

    const uint32_t lin_vj_size = (uint32_t) output_base_offsets[output_base_offsets.size(0)-1].item<int32_t>();
    const uint32_t ab_p_size = (uint32_t) ab_p_to_a.size(0);

    torch::Tensor output = torch::zeros({lin_vj_size, ab_p_size}, W.options());   // |(l_in, v, j), (ab_p)|

    backward_F_stage_one_cuda(output, W, C, G, Y, R, radii,
                              L_out_list, L_in_list, u_sizes, v_sizes,
                              output_base_offsets, C_offsets, G_base_offsets, R_base_offsets,
                              ab_p_to_a,
                              (uint32_t) l_in_max_net_bound);
    return output;
}


torch::Tensor backward_R(
        torch::Tensor W,
        torch::Tensor C,
        torch::Tensor G,
        torch::Tensor F,
        torch::Tensor Y,
        torch::Tensor radii,
        torch::Tensor L_out_list,
        torch::Tensor L_in_list,
        torch::Tensor u_sizes,
        torch::Tensor v_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor C_offsets,
        torch::Tensor G_base_offsets,
        torch::Tensor F_base_offsets,
        torch::Tensor ab_p_to_a,
        torch::Tensor ab_p_to_b,
        const int32_t l_in_max_net_bound
){
    CHECK_FLOAT_INPUT(W);
    CHECK_FLOAT_INPUT(C);
    CHECK_FLOAT_INPUT(G);
    CHECK_FLOAT_INPUT(F);
    CHECK_FLOAT_INPUT(Y);
    CHECK_FLOAT_INPUT(radii);
    CHECK_INT_INPUT(L_out_list);
    CHECK_INT_INPUT(L_in_list);
    CHECK_INT_INPUT(u_sizes);
    CHECK_INT_INPUT(v_sizes);
    CHECK_INT_INPUT(output_base_offsets);
    CHECK_INT_INPUT(C_offsets);
    CHECK_INT_INPUT(G_base_offsets);
    CHECK_INT_INPUT(F_base_offsets);
    CHECK_INT_INPUT(ab_p_to_a);
    CHECK_INT_INPUT(ab_p_to_b);

    const uint32_t lout_lin_luv = (uint32_t) output_base_offsets[output_base_offsets.size(0)-1].item<int32_t>();
    const uint32_t ab_p_size    = (uint32_t) ab_p_to_a.size(0);

    torch::Tensor output = torch::zeros({lout_lin_luv, ab_p_size}, W.options());   // |(l_out, l_in, l, u, v), (ab_p)|

    backward_R_cuda(output, W, C, G, F, Y, radii,
                    L_out_list, L_in_list, u_sizes, v_sizes,
                    output_base_offsets, C_offsets, G_base_offsets, F_base_offsets,
                    ab_p_to_a, ab_p_to_b,
                    (uint32_t) l_in_max_net_bound);
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_stage_one", &forward_stage_one, "Periodic convolution combined with Kernel, forward pass: first stage (CUDA)");
  m.def("backward_F_stage_one", &backward_F_stage_one, "Periodic convolution combined with Kernel, backward pass for features: first stage (CUDA)");
  m.def("backward_R", &backward_R, "Periodic convolution combined with Kernel, forward pass for Radial Basis Function outputs (CUDA)");
}