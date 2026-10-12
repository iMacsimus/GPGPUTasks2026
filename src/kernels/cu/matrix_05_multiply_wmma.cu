#include <memory>

#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
// Если строка "using namespace nvcuda;" не компилируется, добавьте в CMake options: -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo
#include <mma.h>
using namespace nvcuda;

__global__ void compress_a(
                       const float* a, // rows=h x cols=k
                       half *a_compressed,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ half local_a[16][17];

    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;
    const unsigned int r = threadIdx.y;
    const unsigned int c = threadIdx.x;

    const unsigned int row = by * 16 + r;
    const unsigned int col = bx * 16 + c;

    local_a[r][c] = (row < h && col < k) ? __float2half(a[row * k + col]) : __float2half(0.0f);

    __syncthreads();

    const unsigned int k_tiles = (k + 15) / 16;
    const unsigned int tile_index = by * k_tiles + bx;

    a_compressed[tile_index * 256 + r * 16 + c] = local_a[r][c];
}

__global__ void compress_b(
                       const float* b, // rows=k x cols=w
                       half *b_compressed,
                       unsigned int k,
                       unsigned int w)
{
    __shared__ half local_b[16][17];

    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;
    const unsigned int r = threadIdx.y;
    const unsigned int c = threadIdx.x;

    const unsigned int row = by * 16 + r;
    const unsigned int col = bx * 16 + c;

    local_b[r][c] = (row < k && col < w) ? __float2half(b[row * w + col]) : __float2half(0.0f);

    __syncthreads();

    const unsigned int w_tiles = (w + 15) / 16;
    const unsigned int tile_index = by * w_tiles + bx;

    b_compressed[tile_index * 256 + r * 16 + c] = local_b[r][c];
}

__global__ void matrix_multiply_wmma(
                       const half* a_compressed, // rows=h x cols=k, tile-localized
                       const half* b_compressed, // rows=k x cols=w, tile-localized
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ half local_a[8][K_TILES][256];
    __shared__ half local_b[K_TILES][X_TILES][256];

    const uint32_t wi = threadIdx.y;
    const uint32_t tid = threadIdx.y * 32 + threadIdx.x;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[X_TILES];
    for (int j = 0; j < X_TILES; ++j) {
      wmma::fill_fragment(c_frag[j], 0.0f);
    }

    const unsigned int k_tiles_count = (k + 15) / 16;
    const unsigned int w_tiles_count = (w + 15) / 16;

    const half2* a2 = reinterpret_cast<const half2*>(a_compressed);
    const half2* b2 = reinterpret_cast<const half2*>(b_compressed);
    half2* local_a2 = reinterpret_cast<half2*>(&local_a[0][0][0]);
    half2* local_b2 = reinterpret_cast<half2*>(&local_b[0][0][0]);

    const unsigned int a_row_half2 = K_TILES * 128;
    const unsigned int b_row_half2 = X_TILES * 128;

    for (int i = 0; i < k_tiles_count / K_TILES; ++i) {
      __syncthreads();
      for (unsigned int idx = tid; idx < 8 * a_row_half2; idx += 256) {
        const unsigned int r = idx / a_row_half2;
        const unsigned int off = idx % a_row_half2;
        local_a2[idx] = a2[((blockIdx.y * 8 + r) * k_tiles_count + i * K_TILES) * 128 + off];
      }
      for (unsigned int idx = tid; idx < K_TILES * b_row_half2; idx += 256) {
        const unsigned int t = idx / b_row_half2;
        const unsigned int off = idx % b_row_half2;
        local_b2[idx] = b2[((i * K_TILES + t) * w_tiles_count + blockIdx.x * X_TILES) * 128 + off];
      }
      __syncthreads();
      for (int t = 0; t < K_TILES; ++t) {
        wmma::load_matrix_sync(a_frag, local_a[wi][t], 16);
        for (int j = 0; j < X_TILES; ++j) {
          wmma::load_matrix_sync(b_frag, local_b[t][j], 16);
          wmma::mma_sync(c_frag[j], a_frag, b_frag, c_frag[j]);
        }
      }
    }

    uint32_t warpOffsetX = blockIdx.x * (X_TILES * 16);
    uint32_t warpOffsetY = blockIdx.y * (128) + wi * 16;
    for (int j = 0; j < X_TILES; ++j) {
      wmma::store_matrix_sync(c + (warpOffsetY*w + warpOffsetX + j*16), c_frag[j], w, wmma::mem_row_major);
    }
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    static std::shared_ptr<half> a_compressed;
    if (!a_compressed) {
        half *ptr = nullptr;
        cudaMalloc(&ptr, (size_t)((h + 15) / 16) * ((k + 15) / 16) * 256 * sizeof(half));
        a_compressed = { ptr, cudaFree };
    }

    static std::shared_ptr<half> b_compressed;
    if (!b_compressed) {
        half *ptr = nullptr;
        cudaMalloc(&ptr, (size_t)((k + 15) / 16) * ((w + 15) / 16) * 256 * sizeof(half));
        b_compressed = { ptr, cudaFree };
    }

    dim3 compress_block(16, 16);
    dim3 compress_a_grid((k + 15) / 16, (h + 15) / 16);
    ::compress_a<<<compress_a_grid, compress_block, 0, stream>>>(a.cuptr(), a_compressed.get(), h, k);

    dim3 compress_b_grid((w + 15) / 16, (k + 15) / 16);
    ::compress_b<<<compress_b_grid, compress_block, 0, stream>>>(b.cuptr(), b_compressed.get(), k, w);

    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a_compressed.get(), b_compressed.get(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
