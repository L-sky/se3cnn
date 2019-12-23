import torch.nn as nn


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