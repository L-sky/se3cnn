#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#define CUDA_CHECK(stmt) do {                              \
    cudaError_t err = stmt;                                \
    if (err != cudaSuccess) {                              \
      std::cerr << "CUDA failure in " << #stmt << std::endl\
                << cudaGetErrorString(err) << std::endl;   \
      throw std::runtime_error(#stmt);                     \
    }                                                      \
  } while(0)


__device__ constexpr uint32_t threads_per_block_backward_R_cuda_parent_kernel() { return 256; }

template<typename T>
__global__ void backward_R_child_cuda_kernel(
			  T* 	    const __restrict__ output_lout_lin,
		const T* 	    const __restrict__ C_lout_lin,
		const T* 	    const __restrict__ G_lout,
		const T* 	    const __restrict__ F_lin,
		const T* 	    const __restrict__ Y,
		const T* 	    const __restrict__ radii,
		const uint32_t* const __restrict__ ab_p_to_a,
		const uint32_t* const __restrict__ ab_p_to_b,
		const T 					       W_lout_lin_r_nonzero,
		const T 					       W_lout_lin_r_zero,
		const uint32_t					   l_offset,
		const uint32_t					   u_size,
		const uint32_t					   v_size,
		const uint32_t					   ab_p_size,
		const uint32_t					   a_size,
		const uint32_t 					   i_max,
		const uint32_t 					   j_max
){
	const uint32_t uvab_p = threadIdx.x + blockIdx.x * blockDim.x;

	// last block can be incompletely filled, because uvab_p_size is not necessary divisible by set number of threads
	if (blockIdx.x == gridDim.x - 1 && uvab_p >= u_size * v_size * ab_p_size) return;

	const uint32_t l_f   = blockIdx.y + l_offset;
	const uint32_t m_max = 2*l_f + 1;

	// deduce individual indices
	const uint32_t u	= uvab_p % (v_size * ab_p_size);
	const uint32_t v 	= (uvab_p - u * v_size * ab_p_size) % ab_p_size;
	const uint32_t ab_p = (uvab_p - (u * v_size + v) * ab_p_size) / ab_p_size;
	const uint32_t a    = ab_p_to_a[ab_p];
	const uint32_t b 	= ab_p_to_b[ab_p];

	const T norm 	= W_lout_lin_r_nonzero + (radii[ab_p] != 0.) * (W_lout_lin_r_zero - W_lout_lin_r_nonzero);

	// add offsets
	const T* const __restrict__ C_lout_lin_l 	= C_lout_lin 	+ (i_max * j_max * (l_f*l_f - l_offset*l_offset)); 	// only valid L's, thus index is shifted
	const T* const __restrict__ G_lout_u		= G_lout 		+ (u * i_max * a_size);
	const T* const __restrict__ F_lin_v		    = F_lin 		+ (v * j_max * a_size);
	const T* const __restrict__ Y_l 			= Y 			+ (l_f * l_f * ab_p_size);							// contains values without gaps along L

	// make additions (writes) to register
	T output_lout_lin_l_uvab_p = 0;

	for (size_t i = 0; i < i_max; i++){
		for (size_t j = 0; j < j_max; j++){
			for (size_t m = 0; m < m_max; m++){
				// TODO: store repeating values on different levels to reduce number of calls to global memory
				output_lout_lin_l_uvab_p = C_lout_lin[(i*j_max + j)*m_max + m] * G_lout_u[i*a_size + a] * F_lin_v[j*a_size + b] * Y_l[m*ab_p_size + ab_p];
			}
		}
	}

	// write normalized result to global memory
	// blockIdx.y instead of l_f is intentional, we need consequent zero-based index of l here (not actual value)
	output_lout_lin[blockIdx.y * u_size * v_size * ab_p_size + uvab_p] = norm * output_lout_lin_l_uvab_p;
}


template<typename T>
__global__ void backward_R_parent_cuda_kernel(
			  T* 	    const __restrict__ output,		        // placeholder to store gradients
		const T* 	    const __restrict__ W,			        // normalization coefficients
		const T* 	    const __restrict__ C,			        // Clebsch-Gordan coefficients
		const T* 	    const __restrict__ G,			        // gradients coming from next layer
		const T* 	    const __restrict__ F,			        // input features
		const T* 	    const __restrict__ Y,			        // spherical harmonics
		const T* 	    const __restrict__ radii,		        // absolute distances between points and their neighbors
		const uint32_t* const __restrict__ L_out_list,			// output rotational orders
		const uint32_t* const __restrict__ L_in_list,			// input rotational orders
		const uint32_t* const __restrict__ u_sizes,				// output multiplicities
		const uint32_t* const __restrict__ v_sizes,				// input multiplicities
		const uint32_t* const __restrict__ output_base_offsets,	// jump points for indexing output over l_out, l_in
		const uint32_t* const __restrict__ G_base_offsets,		// jump points for indexing G over l_out
		const uint32_t* const __restrict__ C_offsets,			// jump points for indexing C over l_out, l_in
		const uint32_t* const __restrict__ F_base_offsets, 		// jump points for indexing F over l_in
		const uint32_t* const __restrict__ ab_p_to_a,			// map from composite index ab_p to a
		const uint32_t* const __restrict__ ab_p_to_b,			// map from composite index ab_p to the only b holding non-zero value (contraction of the sum along b)
		const uint32_t					   ab_p_size,			// total number of pairs point-neighbor
		const uint32_t					   a_size,				// number of points (atoms)
		const uint32_t 					   l_in_max_net			// maximal value of l_in that is present in C (for selecting offset)
) {
	const uint32_t l_out_id  = blockIdx.x;
	const uint32_t l_in_id 	 = blockIdx.y;
	const uint32_t l_in_size = gridDim.y;

	const uint32_t l_out = L_out_list[l_out_id];
	const uint32_t l_in  = L_in_list[l_in_id];
	const uint32_t i_max = 2*l_out + 1;
	const uint32_t j_max = 2*l_in + 1;

	const uint32_t u_size = u_sizes[l_out_id]; 	// output multiplicity (for particular l_out)
	const uint32_t v_size = v_sizes[l_in_id]; 	// input multiplicity  (for particular l_in)

	/*
	  Expected order of indices:
	 	 output -> [l_out, l_in, l, u, v, a, b_p]
	 	 W 		-> [l_out, l_in, 2]
	 	 C		-> [l_out, l_in, l, i, j, m]
	 	 G 		-> [l_out, u, i, a]
	 	 F 		-> [l_in, v, j, a] 					- here indexing over the last index would be in order defined by ab_p_to_b, but cardinality is a
	 	 Y 		-> [l, m, a, b_p]
	 */
	// add offsets
		  T* const __restrict__ output_lout_lin	= output + (output_base_offsets[l_out_id*l_in_size + l_in_id] * ab_p_size);
	const T* const __restrict__ W_lout_lin		= W + (l_out_id * l_in_size + l_in_id) * 2;
	const T* const __restrict__ C_lout_lin		= C + C_offsets[l_out_id*l_in_max_net + l_in_id]; // TODO: change l_in_max_net + 1 or change to cardinality on prev wrapper
	const T* const __restrict__ G_lout			= G + (G_base_offsets[l_out_id] * a_size);
	const T* const __restrict__ F_lin			= F + (F_base_offsets[l_in_id] * a_size);

	const T W_lout_lin_r_zero    = W_lout_lin[0];
	const T W_lout_lin_r_nonzero = W_lout_lin[1];

	const uint32_t l_offset = abs((int32_t)l_out - (int32_t)l_in);

	const uint32_t threads_per_block = threads_per_block_backward_R_cuda_parent_kernel();
	const uint32_t uvab_p_size = u_size * v_size * ab_p_size;

	dim3 blocks((uvab_p_size + threads_per_block - 1)/threads_per_block, 2*min(l_out, l_in)+1);

	// TODO: for parity we will need to pass additional list with l filters, or maybe recreate get_l_filters_with_parity here
	backward_R_child_cuda_kernel<<<blocks, threads_per_block>>>(output_lout_lin, C_lout_lin, G_lout, F_lin, Y, radii, ab_p_to_a, ab_p_to_b,
			W_lout_lin_r_nonzero, W_lout_lin_r_zero, l_offset, u_size, v_size, ab_p_size, a_size, i_max, j_max);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
}


void backward_R_cuda(
        torch::Tensor output,		        // allocated in higher level wrapper
		torch::Tensor W,                    // layer specific, stored in Hub
		torch::Tensor C,					// object is network wide (sampling is layer specific), stored in Hub
		torch::Tensor G,					// passed by pipeline backward pipeline
		torch::Tensor F,					// layer specific, stored in buffer for backward pass during forward pass
		torch::Tensor Y,					// object is network wide (sampling is layer specific), stored in Hub
		torch::Tensor radii,				// network wide, stored in Hub
		torch::Tensor L_out_list,			// layer specific, stored in Hub
		torch::Tensor L_in_list,			// layer specific, stored in Hub
		torch::Tensor u_sizes,				// layer specific, stored in Hub
		torch::Tensor v_sizes,				// layer specific, stored in Hub
		torch::Tensor output_base_offsets,	// network wide, stored in Hub
		torch::Tensor G_base_offsets,		// layer specific, stored in Hub
		torch::Tensor C_offsets,			// network wide, stored in Hub
		torch::Tensor F_base_offsets, 		// layer specific, stored in Hub
		torch::Tensor ab_p_to_a,			// network wide, stored in Hub
		torch::Tensor ab_p_to_b,			// network wide, stored in Hub
		const uint32_t l_in_max_net			// network wide, stored in Hub // TODO: check if it can be deduced from one of tensor shapes
        ) {

    // const uint32_t ab_p_size,            // deduce from ab_p_to_a.size
    // const uint32_t a_size,			    // deduce from features.size

    // TODO: write kernel call
    /*
    const size_t filters    = Ys.size(0);
    const size_t batch_size = radii.size(0);

    const size_t threads_per_block = 32;                                                    // warp size in contemporary GPUs is 32 threads, this variable should be a multiple of warp size
    dim3 numBlocks((batch_size + threads_per_block - 1)/threads_per_block, filters, 1);     // batch_size/threads_per_block is fractional in general case - round it up

    if (radii.dtype() == torch::kFloat64) {
        rsh_cuda_kernel<double><<<numBlocks, threads_per_block>>>(
            (const double*) radii.data_ptr(), (double*) Ys.data_ptr(), batch_size
        );
    }
    else {                                                                                  // check in C++ binding guarantee that data type is either double (float64) or float (float32)
        rsh_cuda_kernel<float><<<numBlocks, threads_per_block>>>(
            (const float*) radii.data_ptr(), (float*) Ys.data_ptr(), batch_size
        );
    }
    */
}
