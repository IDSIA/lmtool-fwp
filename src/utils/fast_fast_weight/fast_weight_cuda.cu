//
// Original code from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/causal_product/causal_product_cuda.cu
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//
// Modified to implement our update rule; see Sec. 4.2 of our paper.
// Copyright (c) 2021 Kazuki Irie

#include <iostream>
#include <torch/extension.h>
// #include <c10/cuda/CUDAGuard.h>

typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> float_accessor;


__global__ void fast_weight_forward_kernel(
    const float_accessor queries,
    const float_accessor keys,
    const float_accessor values,
    const float_accessor betas,
    float_accessor values_old,
    float_accessor values_insert,
    float_accessor kv,
    float_accessor result,
    const int N,
    const int H,
    const int L,
    const int E,
    const int M,
    const int E_per_block,
    const int blocks_per_sequence,
    const int T,  // block chunk size in time dim.
    const int l_offset  // multiple of T, length offset.
) {
    // Convert indices to the absolute indices
    const int sequence_index = blockIdx.x / blocks_per_sequence;  // seq_index = n * H + h
    int n = sequence_index / H;  // batch id
    int h = sequence_index % H;  // head id

    int e_local = threadIdx.x / M;
    int e_start = ((blockIdx.x % blocks_per_sequence) * E_per_block);
    int e = e_start + e_local;  // id in query dim
    int m = threadIdx.x % M;  // id in value dim

    // Load the shared memory for KV
    const int shared_kv_size = E_per_block * M;
    extern __shared__ float shared_mem[];
    float* shared_kv = shared_mem;  // size: shared_kv_size
    float* shared_results = shared_mem + shared_kv_size;  // size: M
    float* shared_values = shared_results + M;  // size: M*T
    float* shared_keys = shared_values + M*T;  // size: E_per_block*T
    float* shared_queries = shared_keys + E_per_block*T;  // same size as keys.
    float* shared_values_old = shared_queries + E_per_block*T;
    float* shared_values_insert = shared_values_old + M;
    float* shared_betas = shared_values_insert + M;

    if (threadIdx.x < M) {
        shared_results[threadIdx.x] = 0.0;
        shared_values_old[threadIdx.x] = 0.0;
        shared_values_insert[threadIdx.x] = 0.0;
    }

    int t_end = (T + l_offset) <= L ? T : L - l_offset;  // the last segment is shorter.

    for (int i = threadIdx.x; i < (t_end*M); i += blockDim.x)
    {
        int t = int(i / M) + l_offset;
        int d = i % M;
        shared_values[i] = values[n][h][t][d];
    }
    for (int i = threadIdx.x; i < (t_end*E_per_block); i += blockDim.x)
    {
        int t = int(i / E_per_block) + l_offset;
        int d = (i % E_per_block) + e_start;
        if (d < E) {
            shared_keys[i] = keys[n][h][t][d];
            shared_queries[i] = queries[n][h][t][d];
        }
    }
    for (int i = threadIdx.x; i < t_end; i += blockDim.x)
    {
        int t = i + l_offset;
        shared_betas[i] = betas[n][h][t][0];
    }

    __syncthreads();

    if ((n >= N) || (e >= E)) {
        return;
    }

    shared_kv[threadIdx.x] = kv[n][h][e][m];
    for (int t=0; t<t_end; t++) {  // loop over time in the segment
        int l = t + l_offset;  // absolute position in time

        // get old value
        float v_old = shared_kv[e_local*M + m] * shared_keys[t*E_per_block + e_local];
        __syncthreads();

        atomicAdd(
            &shared_values_old[m],
            v_old
        );
        __syncthreads();

        // compute new value to be inserted
        shared_values_insert[m] = shared_betas[t] * (shared_values[t*M + m] - shared_values_old[m]);
        __syncthreads();      
        // Update fast weights
        shared_kv[e_local*M + m] += shared_keys[t*E_per_block + e_local] * shared_values_insert[m];
        __syncthreads();

        float res = shared_queries[t*E_per_block + e_local] * shared_kv[e_local*M + m];

        atomicAdd(
            &shared_results[m],
            res
        );

        __syncthreads();

        if (threadIdx.x < M) {
            float r1 = shared_results[threadIdx.x];
            atomicAdd(
                &result[n][h][l][m],
                r1
            );
            shared_results[threadIdx.x] = 0.0;

            // same for v_old and v_insert
            float r2 = shared_values_old[threadIdx.x];
            atomicAdd(
                &values_old[n][h][l][m],
                r2
            );
            shared_values_old[threadIdx.x] = 0.0;

            float r3 = shared_values_insert[threadIdx.x];
            atomicAdd(
                &values_insert[n][h][l][m],
                r3
            );
            shared_values_insert[threadIdx.x] = 0.0;
        }
    }

    __syncthreads();

    kv[n][h][e][m] = shared_kv[e_local*M + m];  // write back to kv to be carried over to the next segment.
}

void fast_weight_forward(
    const torch::Tensor queries,
    const torch::Tensor keys,
    const torch::Tensor values,
    const torch::Tensor betas,
    torch::Tensor values_old,
    torch::Tensor values_insert,
    torch::Tensor kv,
    torch::Tensor product
) {

//    const at::cuda::OptionalCUDAGuard device_guard(device_of(queries));
    int N = queries.size(0);
    int H = queries.size(1);
    int L = queries.size(2);
    int E = queries.size(3);
    int M = values.size(3);

//    auto kv = torch::zeros({N, H, E, M}, queries.options());

    int threads = 1024;

    // Shared mem max size is 48KB
    int MUL_PER_BLOCK = min(threads, E*M);
    // make sure that MUL_PER_BLOCK is divisible by M;
    MUL_PER_BLOCK = int(MUL_PER_BLOCK / M) *  M;
    threads = MUL_PER_BLOCK;
    const int blocks_per_sequence = ((E*M) + threads -1) / threads;

    const int E_per_block = MUL_PER_BLOCK / M;
    int blocks  = N*H*blocks_per_sequence;

    int shared_mem_const = (E_per_block + 1 + 2)*M;  // KV fast weight, and +1 output/results, + 2 for insert/old value.
    int shared_mem_per_time = (M + 2*E_per_block + 1);  // M for value, 2 * E for query and key, 1 for beta.

    // T = max time chunk size we can afford
    // 12 * 1024 * 4 (float) = 49KB
    const int T = int(((12 * 1024) - shared_mem_const) / shared_mem_per_time);
    const int shared_mem_forward = ((T*shared_mem_per_time) + shared_mem_const) * sizeof(float);

    for (int l_offset=0; l_offset < L; l_offset += T) {
     fast_weight_forward_kernel
            <<<blocks, MUL_PER_BLOCK, shared_mem_forward>>>(
            queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            betas.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values_old.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values_insert.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            kv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            product.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            N, H, L, E, M, E_per_block, blocks_per_sequence, T, l_offset
        );
    }
}



// we need shared memory to store
// Forward direction
// keys, values, gradout
// kv, results
// Backward direction
// queries, gradout, values
// kv_backwards, results
// Shared memory usage
// Forward
// keys: E*T, (values, gradout): M_per_block*T, kv:E*M_per_block, results:E
// Backward
// queries: E*T, (values, gradout): M_per_block*T, kv:E*M_per_block, results:E
// Total memory:
__global__ void fast_weight_backward_query_key_kernel(
    const float_accessor queries,
    const float_accessor keys,
    const float_accessor values,
    const float_accessor betas,
    const float_accessor values_old,
    const float_accessor values_insert,
    const float_accessor grad_out,
    const float_accessor tmp_grad,  // gradient of (-beta * v_old)
    float_accessor kv,
    float_accessor kv_backwards,
    float_accessor grad_queries,
    float_accessor grad_keys,
    int N,
    int H,
    int L,
    int E,
    int M,
    const int M_per_block,
    const int blocks_per_sequence,
    const int T,
    const int l_offset
) {

    const int sequence_index = blockIdx.x / blocks_per_sequence;
    int n = sequence_index / H;
    int h = sequence_index % H;

    int m_local = threadIdx.x / E;
    int m_start = ((blockIdx.x % blocks_per_sequence)*M_per_block);
    int m = m_start + m_local;
    int e = threadIdx.x % E;

    // Load the shared memory
    // Forward memory
    // keys: E*T, (values, gradout): M_per_block*T, kv:E*M_per_block, results:E
    // Backward memory
    // queries: E*T, (values, gradout): M_per_block*T, kv:E*M_per_block, results:E
    // Load the shared memory for KV
    extern __shared__ float shared_mem[];
    const int shared_kv_size = M_per_block * E;
    float* shared_kv = shared_mem;
    float* shared_kv_bw = shared_mem + shared_kv_size;
    float* shared_results = shared_kv_bw + shared_kv_size;
    float* shared_results_bw = shared_results + E;

//    float* shared_keys = shared_results_bw + E;
//    float* shared_values = shared_keys + E*T;
//    float* shared_gradout = shared_values + M_per_block*T;

    float* shared_keys_bw = shared_results_bw + E; 
    float* shared_queries_bw = shared_keys_bw + E*T;
    float* shared_values_bw = shared_queries_bw + E*T;
    float* shared_values_old_bw = shared_values_bw + M_per_block*T;
    float* shared_values_insert_bw = shared_values_old_bw + M_per_block*T;
    float* shared_betas_bw = shared_values_insert_bw + M_per_block*T;
    float* shared_tmp_grad_bw = shared_betas_bw + T;
    float* shared_gradout_bw = shared_tmp_grad_bw + M_per_block*T;

    if (threadIdx.x < E) {
        shared_results[threadIdx.x] = 0.0;
        shared_results_bw[threadIdx.x] = 0.0;
    }

    int t_end = (T + l_offset) <= L ? T : (L - l_offset);
    for (int i = threadIdx.x; i < (t_end*M_per_block); i += blockDim.x)
    {
        int t = int(i / M_per_block) + l_offset;
        int t_bw = L - t - 1;
        int d = (i % M_per_block) + m_start;
        if (d < M) {
            shared_values_bw[i] = values[n][h][t_bw][d];
            shared_values_old_bw[i] = values_old[n][h][t_bw][d];
            shared_values_insert_bw[i] = values_insert[n][h][t_bw][d];
            shared_tmp_grad_bw[i] = tmp_grad[n][h][t_bw][d];
            shared_gradout_bw[i] = grad_out[n][h][t_bw][d];
        }
    }
    for (int i = threadIdx.x; i < (t_end*E); i += blockDim.x)
    {
        int t = int(i / E) + l_offset;
        int t_bw = L - t - 1;
        int d = (i % E);
        shared_keys_bw[i] = keys[n][h][t_bw][d];
        shared_queries_bw[i] = queries[n][h][t_bw][d];
    }
    for (int i = threadIdx.x; i < t_end; i += blockDim.x)
    {
        int t = i + l_offset;
        int t_bw = L - 1 - t;
        shared_betas_bw[i] = betas[n][h][t_bw][0];
    }
    __syncthreads();

    if ((n >= N) || (m >= M)) {
        return;
    }

    shared_kv[threadIdx.x] = kv[n][h][e][m];
    shared_kv_bw[threadIdx.x] = kv_backwards[n][h][e][m];

    for (int t=0; t<t_end; t++) {
        int l = t + l_offset;
        int l_b = L - l -1;

        shared_kv_bw[m_local*E + e] += shared_queries_bw[t*E + e] * shared_gradout_bw[t*M_per_block + m_local];
        __syncthreads();

        float res = shared_gradout_bw[t*M_per_block + m_local] * shared_kv[m_local*E + e];  // grad_q
        float res_bw = (shared_values_bw[t*M_per_block + m_local] - shared_values_old_bw[t*M_per_block + m_local]) * shared_kv_bw[m_local*E + e] * shared_betas_bw[t];
        atomicAdd(
            &shared_results[e],
            res
        );
        atomicAdd(
            &shared_results_bw[e],
            res_bw
        );  // key grad part 1 and 2 of 3.
        // end query
        // grad key part 3 of 3
        // reverse update FWM
        shared_kv[m_local*E + e] -= shared_keys_bw[t*E + e] * shared_values_insert_bw[t*M_per_block + m_local];  // key bw, insert bw
        __syncthreads();

        float res_k = shared_kv[m_local*E + e] * shared_tmp_grad_bw[t*M_per_block + m_local];
        atomicAdd(
            &shared_results_bw[e],
            res_k
        );
        __syncthreads();
        if (threadIdx.x < E) {
            float rk2 = shared_results_bw[threadIdx.x];
            atomicAdd(
                &grad_keys[n][h][l_b][e],
                rk2
            );
            shared_results_bw[threadIdx.x] = 0.0;
        } 
        if (threadIdx.x < E) {
            float rq = shared_results[threadIdx.x];
            float rk = shared_results_bw[threadIdx.x];
            atomicAdd(
                &grad_queries[n][h][l_b][e],
                rq
            );
            atomicAdd(
                &grad_keys[n][h][l_b][e],
                rk
            );
            shared_results[threadIdx.x] = 0.0;
            shared_results_bw[threadIdx.x] = 0.0;
        }
        // remainder grad for fwm
        shared_kv_bw[m_local*E + e] += shared_keys_bw[t*E + e] * shared_tmp_grad_bw[t*M_per_block + m_local];
    }
    __syncthreads();
    kv[n][h][e][m] = shared_kv[m_local*E + e];
    kv_backwards[n][h][e][m] = shared_kv_bw[m_local*E + e];
}


__global__ void fast_weight_backward_value_beta_kernel(
    const float_accessor queries,
    const float_accessor keys,
    const float_accessor values,
    const float_accessor betas,
    const float_accessor values_old,
    const float_accessor grad_out,
    float_accessor kv,
    float_accessor grad_values,
    float_accessor grad_betas,
    float_accessor tmp_grad,
    int N,
    int H,
    int L,
    int E,
    int M,
    int E_per_block,
    int blocks_per_sequence,
    int T,
    int l_offset
) {

    const int sequence_index = blockIdx.x / blocks_per_sequence;
    int n = sequence_index / H;
    int h = sequence_index % H;

    int e_local = threadIdx.x / M;
    int e_start = ((blockIdx.x % blocks_per_sequence) * E_per_block);
    int e = e_start + e_local;
    int m = threadIdx.x % M;

    // Load the shared memory for KV
    const int shared_kv_size = E_per_block * M;
    extern __shared__ float shared_mem[];
    float* shared_kv = shared_mem;
    float* shared_results = shared_mem + shared_kv_size;
    float* shared_results_beta = shared_results + M;

    float* shared_gradout = shared_results_beta + 1;
    float* shared_keys = shared_gradout + M*T;
    float* shared_queries = shared_keys + E_per_block*T;
    float* shared_betas = shared_queries + E_per_block*T;
    float* shared_values_old = shared_betas + T;
    float* shared_values = shared_values_old + M*T;

    if (threadIdx.x < M) {
        shared_results[threadIdx.x] = 0.0;
    }
    if (threadIdx.x < 1) {
        shared_results_beta[threadIdx.x] = 0.0;
    }
    // Everythig goes backward
    int t_end = (T + l_offset) <= L ? T : L - l_offset;
    for (int i = threadIdx.x; i < (t_end*M); i += blockDim.x)
    {
        int t = int(i / M) + l_offset;
        int t_bw = L - 1 - t;
        int d = i % M;
        shared_gradout[i] = grad_out[n][h][t_bw][d];
        shared_values[i] = values[n][h][t_bw][d];
        shared_values_old[i] = values_old[n][h][t_bw][d];
    }
    for (int i = threadIdx.x; i < (t_end*E_per_block); i += blockDim.x)
    {
        int t = int(i / E_per_block) + l_offset;
        int t_bw = L - 1 - t;
        int d = (i % E_per_block) + e_start;
        if (d < E) {
            shared_keys[i] = keys[n][h][t_bw][d];
            shared_queries[i] = queries[n][h][t_bw][d];
        }
    }
    for (int i = threadIdx.x; i < t_end; i += blockDim.x)
    {
        int t = i + l_offset;
        int t_bw = L - 1 - t;
        shared_betas[i] = betas[n][h][t_bw][0];
    }
    __syncthreads();
    if ((n >= N) || (e >= E)){
        return;
    }
    
    shared_kv[threadIdx.x] = kv[n][h][e][m];
    for (int t=0; t<t_end; t++) {
        int l = t + l_offset;
        int l_b = L - l -1;

        shared_kv[e_local*M + m] += shared_queries[t*E_per_block + e_local] * shared_gradout[t*M + m];
        __syncthreads();

        float res = shared_keys[t*E_per_block + e_local] * shared_kv[e_local*M + m];
        atomicAdd(
            &shared_results[m],
            res
        );
        __syncthreads();

        shared_kv[e_local*M + m] -= shared_betas[t] * shared_results[m] * shared_keys[t*E_per_block + e_local];
        __syncthreads();

        if (threadIdx.x < M) {
            float r1 = shared_results[threadIdx.x] * shared_betas[t];
            atomicAdd(
                &grad_values[n][h][l_b][m],
                r1
            );

            float r2 = -r1;
            atomicAdd(
                &tmp_grad[n][h][l_b][m],
                r2
            );

            float res_beta = shared_results[threadIdx.x] * (shared_values[t*M + m] - shared_values_old[t*M + m]);
            atomicAdd(
            &shared_results_beta[0],
            res_beta
            );

            shared_results[threadIdx.x] = 0.0;
        }

        if (threadIdx.x < 1) {
            float r3 = shared_results_beta[threadIdx.x];
            atomicAdd(
                &grad_betas[n][h][l_b][0],
                r3
            );
            shared_results_beta[0] = 0.0;
        }
    }
    __syncthreads();
    kv[n][h][e][m] = shared_kv[e_local*M + m];
}


// Backward
void fast_weight_backward(
    const torch::Tensor queries,
    const torch::Tensor keys,
    const torch::Tensor values,
    const torch::Tensor betas,
    const torch::Tensor values_old,
    const torch::Tensor values_insert,
    const torch::Tensor grad_out,
    torch::Tensor fast_weight,
    torch::Tensor grad_queries,
    torch::Tensor grad_keys,
    torch::Tensor grad_values,
    torch::Tensor grad_betas
) {
//    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_queries));
    int N = queries.size(0);
    int H = queries.size(1);
    int L = queries.size(2);
    int E = queries.size(3);
    int M = values.size(3);

    auto kv = torch::zeros({N, H, E, M}, queries.options());
    auto kv_backward = torch::zeros({N, H, E, M}, queries.options());
    auto tmp_grad = torch::zeros({N, H, L, M}, queries.options());

    const int threads = 1024;

    // First part ====================================
    int MPB = min(threads, E*M);
    // make sure that MUL_PER_BLOCK is divisible by M;
    MPB = int(MPB / M) *  M;
    const int blocks_per_sequence_value = ((E*M) + MPB - 1)/ MPB;
    const int E_per_block = MPB / M;
    const int blocks_value = N*H*blocks_per_sequence_value;

    int shared_mem_const = (E_per_block + 1)*M + 1;  // KV (E*M) and output M, +1 for grad_beta
    int shared_mem_per_time = (3*M + 2*E_per_block + 1);  // 3M for value, value_old, and grad_out, 2 * E for query and key, 1 for beta.

    int T = int(((12 * 1024) - shared_mem_const) / shared_mem_per_time);
    const int shared_mem_v_backward = ((T*shared_mem_per_time) + shared_mem_const) * sizeof(float);
    // kv.zero_();
    for (int l_offset=0; l_offset < L; l_offset += T) {
        fast_weight_backward_value_beta_kernel
            <<<blocks_value, MPB, shared_mem_v_backward>>>(
            queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            betas.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values_old.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),  // inserted_values is not needed here.
            grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            kv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_betas.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            tmp_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            N, H, L, E, M, E_per_block, blocks_per_sequence_value, T, l_offset
        );
    }

    // Second part ===================================
    int MUL_PER_BLOCK = min(threads, E*M);
    // make sure that MUL_PER_BLOCK is divisible by E;
    MUL_PER_BLOCK = int(MUL_PER_BLOCK / E) *  E;
    const int blocks_per_sequence = ((E*M) + MUL_PER_BLOCK -1) / MUL_PER_BLOCK;
    const int M_per_block = MUL_PER_BLOCK / E;
    int blocks  = N*H*blocks_per_sequence;

    // Forward memory
    // keys: E*T, (values, gradout): M_per_block*T, kv:E*M_per_block, results:E
    // Backward memory
    // queries: E*T, (values, gradout): M_per_block*T, kv:E*M_per_block, results:E
    // Total memory
    // 2*((E + 2*M_per_block)*T + (E+1)*M_per_block)
    shared_mem_const = 2*E*(1+M_per_block);  // 2*E*M for KV, 2*E outputs key query
    shared_mem_per_time = 2*E + 5*M_per_block + 1;  // 2*E for key query, 5M for value, old, insert, tmp_grad, grad_out.
    T = int(((12 * 1024) - shared_mem_const) / shared_mem_per_time);
    const int shared_mem_qk_backward = ((T*shared_mem_per_time) + shared_mem_const) * sizeof(float);
//    kv.zero_();
    for (int l_offset=0; l_offset < L; l_offset += T) {
        fast_weight_backward_query_key_kernel
            <<<blocks, MUL_PER_BLOCK, shared_mem_qk_backward>>>(
            queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            betas.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values_old.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values_insert.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            tmp_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            fast_weight.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            kv_backward.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            N, H, L, E, M, M_per_block, blocks_per_sequence, T, l_offset
        );
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fast_weight_forward",
        &fast_weight_forward,
        "Compute the weighted sum of values but attending only to previous "
        "values."
    );
    m.def(
        "fast_weight_backward",
        &fast_weight_backward,
        "Compute the gradients for the fast weight memory."
    );
}
