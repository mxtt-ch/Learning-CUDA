#include <vector>
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>

#include "../tester/utils.h"

template <typename T>
__global__ void quickSelectKernel(T* d_data, int n, int k, T* d_result);

template <typename T>
__global__ void matrixMultiplyKernel(const T* Q, const T* K, T* QK, 
                                    int batch_size, int target_seq_len, int src_seq_len,
                                    int query_heads, int kv_heads, int head_dim);

template <typename T>
__global__ void applyCausalMaskKernel(T* QK, int target_seq_len, int src_seq_len, int query_heads);

template <typename T>
__global__ void softmaxKernel(T* QK, T* attention_weights, int target_seq_len, int src_seq_len, int query_heads);


template <typename T>
__global__ void outputKernel(const T* attention_weights, const T* V, T* output,
                            int batch_size, int target_seq_len, int src_seq_len,
                            int query_heads, int kv_heads, int head_dim);

/**
 * @brief Find the k-th largest element in a vector using CUDA.
 * 
 * @tparam T Type of elements in the input vector (should support `int` and `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the largest element).
 * @return T The k-th largest element in `h_input`.

 * @note Must use CUDA kernels for all compute-intensive steps; no significant CPU allowed.
 * @note Library functions that can directly complete a significant part of the work are NOT allowed. 
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally. Errors should be thrown.
 */
template <typename T>
T kthLargest(const std::vector<T>& h_input, size_t k) {
  // TODO: Implement the kthLargest function
    if (h_input.empty() || k == 0 || k > h_input.size()) {
        return T(-100);
    }
    
    int n = h_input.size();

    T *d_data, *d_result;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_data, h_input.data(), n * sizeof(T), cudaMemcpyHostToDevice));

    int blockSize_kth = 256;
    int gridSize_kth = 1;
    quickSelectKernel<T><<<gridSize_kth, blockSize_kth>>>(d_data, n, k, d_result);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    T result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    
    return result;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
    size_t output_size = batch_size * target_seq_len * query_heads * head_dim;
    h_o.resize(output_size);

    T *d_q, *d_k, *d_v, *d_o, *d_qk, *d_attention_weights;
    
    size_t q_size = batch_size * target_seq_len * query_heads * head_dim * sizeof(T);
    size_t k_size = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
    size_t v_size = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
    size_t o_size = output_size * sizeof(T);
    size_t qk_size = batch_size * target_seq_len * query_heads * src_seq_len * sizeof(T);
    size_t attn_size = batch_size * target_seq_len * query_heads * src_seq_len * sizeof(T);
    
    CUDA_CHECK(cudaMalloc(&d_q, q_size));
    CUDA_CHECK(cudaMalloc(&d_k, k_size));
    CUDA_CHECK(cudaMalloc(&d_v, v_size));
    CUDA_CHECK(cudaMalloc(&d_o, o_size));
    CUDA_CHECK(cudaMalloc(&d_qk, qk_size));
    CUDA_CHECK(cudaMalloc(&d_attention_weights, attn_size));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice));

    int blockSize_attn = 256;
    int gridSize_attn = batch_size * target_seq_len * query_heads;
    matrixMultiplyKernel<T><<<gridSize_attn, blockSize_attn>>>(d_q, d_k, d_qk, 
                                                    batch_size, target_seq_len, src_seq_len,
                                                    query_heads, kv_heads, head_dim);

    if (is_causal) {
        int maskGridSize = (batch_size * target_seq_len * query_heads * src_seq_len + blockSize_attn - 1) / blockSize_attn;
        applyCausalMaskKernel<T><<<maskGridSize, blockSize_attn>>>(d_qk, target_seq_len, src_seq_len, query_heads);
    }

    int softmaxGridSize = (batch_size * target_seq_len * query_heads * src_seq_len + blockSize_attn - 1) / blockSize_attn;
    softmaxKernel<T><<<softmaxGridSize, blockSize_attn>>>(d_qk, d_attention_weights, 
                                                    target_seq_len, src_seq_len, query_heads);

    outputKernel<T><<<gridSize_attn, blockSize_attn>>>(d_attention_weights, d_v, d_o,
                                           batch_size, target_seq_len, src_seq_len,
                                           query_heads, kv_heads, head_dim);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));
    CUDA_CHECK(cudaFree(d_qk));
    CUDA_CHECK(cudaFree(d_attention_weights));
}

// CUDA kernel implementations
template <typename T>
__global__ void quickSelectKernel(T* d_data, int n, int k, T* d_result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        int left = 0, right = n - 1;
        while (left <= right) {
            T pivot = d_data[right];
            int i = left - 1;

            for (int j = left; j < right; j++) {
                if (d_data[j] >= pivot) {
                    i++;
                    T temp = d_data[i];
                    d_data[i] = d_data[j];
                    d_data[j] = temp;
                }
            }
            
            T temp = d_data[i + 1];
            d_data[i + 1] = d_data[right];
            d_data[right] = temp;
            
            int pivotIndex = i + 1;
            
            if (pivotIndex == k - 1) {
                d_result[0] = d_data[pivotIndex];
                return;
            } else if (pivotIndex < k - 1) {
                left = pivotIndex + 1;
            } else {
                right = pivotIndex - 1;
            }
        }
        d_result[0] = d_data[k - 1];
    }
}

template <typename T>
__global__ void matrixMultiplyKernel(const T* Q, const T* K, T* QK, 
                                    int batch_size, int target_seq_len, int src_seq_len,
                                    int query_heads, int kv_heads, int head_dim) {
    int batch = blockIdx.x / (target_seq_len * query_heads);
    int seq_idx = (blockIdx.x % (target_seq_len * query_heads)) / query_heads;
    int head_idx = blockIdx.x % query_heads;
    int kv_head_idx = head_idx % kv_heads;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    T sum = 0.0f;
    
    for (int dim = tid; dim < head_dim; dim += stride) {
        int q_idx = batch * target_seq_len * query_heads * head_dim + 
                   seq_idx * query_heads * head_dim + head_idx * head_dim + dim;
        int k_idx = batch * src_seq_len * kv_heads * head_dim + 
                   seq_idx * kv_heads * head_dim + kv_head_idx * head_dim + dim;
        
        sum += Q[q_idx] * K[k_idx];
    }

    for (int offset = stride / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (tid == 0) {
        int out_idx = batch * target_seq_len * query_heads * src_seq_len + 
                     seq_idx * query_heads * src_seq_len + head_idx * src_seq_len + seq_idx;
        QK[out_idx] = sum;
    }
}

template <typename T>
__global__ void applyCausalMaskKernel(T* QK, int target_seq_len, int src_seq_len, int query_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = target_seq_len * src_seq_len * query_heads;
    
    if (idx < total_elements) {
        int seq_idx = idx / (src_seq_len * query_heads);
        int src_idx = (idx % (src_seq_len * query_heads)) / query_heads;
        
        if (src_idx > seq_idx) {
            QK[idx] = -1e9f;
        }
    }
}

template <typename T>
__global__ void softmaxKernel(T* QK, T* attention_weights, int target_seq_len, int src_seq_len, int query_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = target_seq_len * src_seq_len * query_heads;
    
    if (idx < total_elements) {
        int seq_idx = idx / (src_seq_len * query_heads);
        int head_idx = (idx % (src_seq_len * query_heads)) % query_heads;

        T max_val = -1e9f;
        for (int i = 0; i < src_seq_len; i++) {
            int temp_idx = seq_idx * src_seq_len * query_heads + i * query_heads + head_idx;
            max_val = (QK[temp_idx] > max_val) ? QK[temp_idx] : max_val;
        }

        T exp_val = expf(QK[idx] - max_val);
        T sum = 0.0f;
        
        for (int i = 0; i < src_seq_len; i++) {
            int temp_idx = seq_idx * src_seq_len * query_heads + i * query_heads + head_idx;
            sum += expf(QK[temp_idx] - max_val);
        }
        
        attention_weights[idx] = exp_val / sum;
    }
}

template <typename T>
__global__ void outputKernel(const T* attention_weights, const T* V, T* output,
                            int batch_size, int target_seq_len, int src_seq_len,
                            int query_heads, int kv_heads, int head_dim) {
    int batch = blockIdx.x / (target_seq_len * query_heads);
    int seq_idx = (blockIdx.x % (target_seq_len * query_heads)) / query_heads;
    int head_idx = blockIdx.x % query_heads;
    int kv_head_idx = head_idx % kv_heads;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    T sum = 0.0f;
    
    for (int dim = tid; dim < head_dim; dim += stride) {
        for (int src = 0; src < src_seq_len; src++) {
            int attn_idx = batch * target_seq_len * query_heads * src_seq_len + 
                          seq_idx * query_heads * src_seq_len + head_idx * src_seq_len + src;
            int v_idx = batch * src_seq_len * kv_heads * head_dim + 
                       src * kv_heads * head_dim + kv_head_idx * head_dim + dim;
            
            sum += attention_weights[attn_idx] * V[v_idx];
        }
    }

    for (int offset = stride / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (tid == 0) {
        int out_idx = batch * target_seq_len * query_heads * head_dim + 
                     seq_idx * query_heads * head_dim + head_idx * head_dim;
        output[out_idx] = sum;
    }
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int>&, size_t);
template float kthLargest<float>(const std::vector<float>&, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
