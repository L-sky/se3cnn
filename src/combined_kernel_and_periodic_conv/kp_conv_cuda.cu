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

__device__ constexpr uint32_t threads_per_block_forward_stage_one_parent_cuda_kernel() { return 256; }
__device__ constexpr uint32_t threads_per_block_backward_F_cuda_kernel() { return 256; }
__device__ constexpr uint32_t threads_per_block_backward_R_parent_cuda_kernel() { return 256; }

template<typename T>
__global__ void forward_stage_one_parent_cuda_kernel(
              T*        const __restrict__ output,
        const T*        const __restrict__ W,
        const T*        const __restrict__ C,
        const T*        const __restrict__ F,
        const T*        const __restrict__ Y,
        const T*        const __restrict__ R,
        const T*        const __restrict__ radii,
        const uint32_t* const __restrict__ L_out_list,
        const uint32_t* const __restrict__ L_in_list,
        const uint32_t* const __restrict__ u_sizes,
        const uint32_t* const __restrict__ v_sizes,
        const uint32_t* const __restrict__ output_base_offsets,
        const uint32_t* const __restrict__ C_offsets,
        const uint32_t* const __restrict__ F_base_offsets,
        const uint32_t* const __restrict__ R_base_offsets,
        const uint32_t* const __restrict__ ab_p_to_b,
        const uint32_t					   ab_p_size,
        const uint32_t 					   l_in_max_net
){
    const uint32_t l_out_id  = blockIdx.x;
	const uint32_t l_in_id 	 = blockIdx.y;
	const uint32_t l_in_size = gridDim.y;

	const uint32_t l_out = L_out_list[l_out_id];
	const uint32_t l_in  = L_in_list[l_in_id];
	const uint32_t i_size = 2*l_out + 1;
	const uint32_t j_size = 2*l_in + 1;

	const uint32_t u_size = u_sizes[l_out_id]; 	// output multiplicity (for particular l_out)
	const uint32_t v_size = v_sizes[l_in_id]; 	// input multiplicity  (for particular l_in)

    /*
	  Expected order of indices:
	 	 output -> [l_out, u, i, a, b_p]
	 	 W 		-> [l_out, l_in, 2]
	 	 C		-> [l_out, l_in, l, i, j, m]
	 	 F 		-> [l_in, v, j, a] 					- here indexing over the last index would be in order defined by ab_p_to_b, but cardinality is a
	 	 Y 		-> [l, m, a, b_p]
	 	 R      -> [l_out, l_in, l, u, v, a, b_p]
	 */
	// add offsets
		  T* const __restrict__ output_lout	    = output + (output_base_offsets[l_out_id] * ab_p_size);             // base offsets are the same as for gradients
	const T* const __restrict__ W_lout_lin		= W + (l_out_id * l_in_size + l_in_id) * 2;
	const T* const __restrict__ C_lout_lin		= C + C_offsets[l_out_id*l_in_max_net + l_in_id];                   // TODO: change l_in_max_net + 1 or change to cardinality on prev wrapper
	const T* const __restrict__ F_lin			= F + (F_base_offsets[l_in_id] * a_size);
	const T* const __restrict__ R_lout_lin      = R + (R_base_offsets[l_out_id*l_in_size + l_in_id] * ab_p_size);


	const T W_lout_lin_r_zero    = W_lout_lin[0];
	const T W_lout_lin_r_nonzero = W_lout_lin[1];

	const uint32_t l_min = abs((int32_t)l_out - (int32_t)l_in);
	const uint32_t l_max = l_out + l_in;

	const uint32_t threads_per_block = threads_per_block_forward_stage_one_parent_cuda_kernel()();
	const uint32_t uiab_p_size = u_size * i_size * ab_p_size;

	dim3 blocks((uiab_p_size + threads_per_block - 1)/threads_per_block, 2*min(l_out, l_in)+1);

    forward_stage_one_child_cuda_kernel<<<blocks, threads_per_block>>>(output_lout, C_lout_lin, F_lin, Y, R_lout_lin, radii, ab_p_to_b,
            W_lout_lin_r_nonzero, W_lout_lin_r_zero, l_min, l_max, u_size, v_size, ab_p_size, i_size, j_size);
}

template<typename T>
__global__ void forward_stage_one_child_cuda_kernel(
              T*        const __restrict__ output_lout,
        const T*        const __restrict__ C_lout_lin,
        const T*        const __restrict__ F_lin,
        const T*        const __restrict__ Y,
        const T*        const __restrict__ R_lout_lin,
        const T*        const __restrict__ radii,
        const uint32_t* const __restrict__ ab_p_to_b,
        const T 					       W_lout_lin_r_nonzero,
		const T 					       W_lout_lin_r_zero,
		const uint32_t					   l_min,
		const uint32_t					   l_max,
		const uint32_t					   u_size,
		const uint32_t					   v_size,
		const uint32_t					   ab_p_size,
		const uint32_t 					   i_size,
		const uint32_t 					   j_size
){
    const uint32_t uiab_p = threadIdx.x + blockIdx.x * blockDim.x;

    // last block can be incompletely filled, because uvab_p_size is not necessary divisible by set number of threads
	if (blockIdx.x == gridDim.x - 1 && uiab_p >= u_size * i_size * ab_p_size) return;

	// deduce individual indices
	const uint32_t u	= uiab_p / (i_size * ab_p_size);
	const uint32_t i 	= (uiab_p - u * i_size * ab_p_size) / ab_p_size;
	const uint32_t ab_p = uiab_p - u * i_size * ab_p_size - i * ab_p_size;
	const uint32_t b 	= ab_p_to_b[ab_p];                                      // Note: b is not b_p, it is an index of the only corresponding non-zero entry

	const T norm = W_lout_lin_r_nonzero + (T) (radii[ab_p] != 0.) * (W_lout_lin_r_zero - W_lout_lin_r_nonzero);

	T output_lout_u_i_ab_p_addendum = 0;

	for(uint32_t v = 0; v < v_size; ++v){
	    for(uint32_t j = 0; j < j_size; ++j){
	        for(uint32_t l_f = l_min, l_id = 0; l_f <= l_max; ++l_f, ++l_id){
	            for(uint32_t m = 0, m_size = 2*l_f + 1; m < m_size; ++m){
	                // TODO: recollect indices and distribute over for loops, maybe change order of loops
	                output_lout_u_i_ab_p_addendum +=
	                    C_lout_lin[(l_f*l_f - l_min*l_min)*i_size*j_size + i*j_size*m_size + j*m_size + m] *
	                    F_lin[v*j_size*a_size + j*a_size + b] *
	                    Y[l_f*l_f*ab_p] *
	                    R_lout_lin[l_id*u_size*v_size*ab_p_size + u*v_size*ab_p_size + v*ab_p_size + ab_p];
	            }
	        }
	    }
	}

	atomicAdd(&output_lout[uiab_p], norm * output_lout_u_i_ab_p_addendum);
}


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
		const uint32_t 					   i_size,
		const uint32_t 					   j_size
){
	const uint32_t uvab_p = threadIdx.x + blockIdx.x * blockDim.x;

	// last block can be incompletely filled, because uvab_p_size is not necessary divisible by set number of threads
	if (blockIdx.x == gridDim.x - 1 && uvab_p >= u_size * v_size * ab_p_size) return;

	const uint32_t l_f    = blockIdx.y + l_offset;
	const uint32_t m_size = 2*l_f + 1;

	// deduce individual indices
	const uint32_t u	= uvab_p / (v_size * ab_p_size);
	const uint32_t v 	= (uvab_p - u * v_size * ab_p_size) / ab_p_size;
	const uint32_t ab_p = uvab_p - u * v_size * ab_p_size - v * ab_p_size ;
	const uint32_t a    = ab_p_to_a[ab_p];
	const uint32_t b 	= ab_p_to_b[ab_p];

	const T norm = W_lout_lin_r_nonzero + (T) (radii[ab_p] != 0.) * (W_lout_lin_r_zero - W_lout_lin_r_nonzero);

	// add offsets
	const T* const __restrict__ C_lout_lin_l 	= C_lout_lin 	+ (i_size * j_size * (l_f*l_f - l_offset*l_offset)); 	// only valid L's, thus index is shifted
	const T* const __restrict__ G_lout_u		= G_lout 		+ (u * i_size * a_size);
	const T* const __restrict__ F_lin_v		    = F_lin 		+ (v * j_size * a_size);
	const T* const __restrict__ Y_l 			= Y 			+ (l_f * l_f * ab_p_size);							// contains values without gaps along L

	// make additions (writes) to register
	T output_lout_lin_l_uvab_p = 0;

    size_t ijm = 0;
	for (size_t i = 0; i < i_size; i++){
		for (size_t j = 0; j < j_size; j++){
			for (size_t m = 0; m < m_size; m++, ijm++){
				// TODO: store repeating values on different levels to reduce number of calls to global memory
				output_lout_lin_l_uvab_p += C_lout_lin[ijm] * G_lout_u[i*a_size + a] * F_lin_v[j*a_size + b] * Y_l[m*ab_p_size + ab_p];
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
	const uint32_t i_size = 2*l_out + 1;
	const uint32_t j_size = 2*l_in + 1;

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
		  T* const __restrict__ output_lout_lin	= output + (output_base_offsets[l_out_id*l_in_size + l_in_id] * ab_p_size); // base offsets are the same as for R
	const T* const __restrict__ W_lout_lin		= W + (l_out_id * l_in_size + l_in_id) * 2;
	const T* const __restrict__ C_lout_lin		= C + C_offsets[l_out_id*l_in_max_net + l_in_id];                           // TODO: change l_in_max_net + 1 or change to cardinality on prev wrapper
	const T* const __restrict__ G_lout			= G + (G_base_offsets[l_out_id] * a_size);
	const T* const __restrict__ F_lin			= F + (F_base_offsets[l_in_id] * a_size);

	const T W_lout_lin_r_zero    = W_lout_lin[0];
	const T W_lout_lin_r_nonzero = W_lout_lin[1];

	const uint32_t l_offset = abs((int32_t)l_out - (int32_t)l_in);

	const uint32_t threads_per_block = threads_per_block_backward_R_parent_cuda_kernel();
	const uint32_t uvab_p_size = u_size * v_size * ab_p_size;

	dim3 blocks((uvab_p_size + threads_per_block - 1)/threads_per_block, 2*min(l_out, l_in)+1);

	// TODO: for parity we will need to pass additional list with l filters, or maybe recreate get_l_filters_with_parity here
	backward_R_child_cuda_kernel<<<blocks, threads_per_block>>>(output_lout_lin, C_lout_lin, G_lout, F_lin, Y, radii, ab_p_to_a, ab_p_to_b,
			W_lout_lin_r_nonzero, W_lout_lin_r_zero, l_offset, u_size, v_size, ab_p_size, a_size, i_size, j_size);

    /*
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
    */
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
