#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define HD __host__ __device__

// 4x4, хранимая ПО СТРОКАМ: v[row*4 + col].
// Конструктор принимает 4 строки (по одному float4 на строку).
struct float4x4 {
  float v[16] = {};

  float4x4() = default;
  HD float4x4(float4 r0, float4 r1, float4 r2, float4 r3) {
    v[0] = r0.x; v[1] = r0.y; v[2] = r0.z; v[3] = r0.w;
    v[4] = r1.x; v[5] = r1.y; v[6] = r1.z; v[7] = r1.w;
    v[8] = r2.x; v[9] = r2.y; v[10]= r2.z; v[11]= r2.w;
    v[12]= r3.x; v[13]= r3.y; v[14]= r3.z; v[15]= r3.w;
  }

  HD float  operator()(int r, int c) const { return v[r*4 + c]; }
  HD float& operator()(int r, int c)       { return v[r*4 + c]; }
};

// acc += a * b для 4x4.
HD void fma4x4(const float4x4 &a, const float4x4 &b, float4x4 &acc) {
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
        acc(i, j) += a(i, kk) * b(kk, j);
}

#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/cuda/cu/common.cu>
#include "helpers/rassert.cu"
#include "../defines.h"

#define HD __host__ __device__

// Скалярный tile с padding:
// 4×4-блок занимает SMEM_STRIDE=17 float вместо 16
#define SMEM_STRIDE 17
#define SMEM_TILE_FLOATS (MATMUL_DIM * MATMUL_DIM * SMEM_STRIDE)

__global__ void matrix_multiply_via_local_memory(
                       const float4* a, // h × k, float4 = 4 подряд идущих элемента строки
                       const float4* b, // k × w
                             float4* c, // h × w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ float a_smem[SMEM_TILE_FLOATS];
    __shared__ float b_smem[SMEM_TILE_FLOATS];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t localX = threadIdx.x;
    uint32_t localY = threadIdx.y;

    uint32_t w4 = w / 4;
    uint32_t h4 = h / 4;
    uint32_t k4 = k / 4;

    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;

    const int my_off = (localY * MATMUL_DIM + localX) * SMEM_STRIDE;

    for (int bi = 0; bi < k4 / MATMUL_DIM; ++bi) {
        // ---------- Загрузка в shared ----------
        if (x < w4 && y < h4) {
            // a: 4 float4 (строки y*4 .. y*4+3, столбцовый блок bi*MATMUL_DIM + localX)
            float4 a0 = a[(y*4 + 0) * k4 + bi * MATMUL_DIM + localX];
            float4 a1 = a[(y*4 + 1) * k4 + bi * MATMUL_DIM + localX];
            float4 a2 = a[(y*4 + 2) * k4 + bi * MATMUL_DIM + localX];
            float4 a3 = a[(y*4 + 3) * k4 + bi * MATMUL_DIM + localX];

            // b: 4 float4 (строки bi*MATMUL_DIM + localY .. +3, столбцовый блок x)
            uint32_t br = (bi * MATMUL_DIM + localY) * 4;
            float4 b0 = b[(br + 0) * w4 + x];
            float4 b1 = b[(br + 1) * w4 + x];
            float4 b2 = b[(br + 2) * w4 + x];
            float4 b3 = b[(br + 3) * w4 + x];

            // Раскладываем по строкам 4×4 блока
            int o = my_off;
            a_smem[o + 0]  = a0.x; a_smem[o + 1]  = a0.y; a_smem[o + 2]  = a0.z; a_smem[o + 3]  = a0.w;
            a_smem[o + 4]  = a1.x; a_smem[o + 5]  = a1.y; a_smem[o + 6]  = a1.z; a_smem[o + 7]  = a1.w;
            a_smem[o + 8]  = a2.x; a_smem[o + 9]  = a2.y; a_smem[o + 10] = a2.z; a_smem[o + 11] = a2.w;
            a_smem[o + 12] = a3.x; a_smem[o + 13] = a3.y; a_smem[o + 14] = a3.z; a_smem[o + 15] = a3.w;

            b_smem[o + 0]  = b0.x; b_smem[o + 1]  = b0.y; b_smem[o + 2]  = b0.z; b_smem[o + 3]  = b0.w;
            b_smem[o + 4]  = b1.x; b_smem[o + 5]  = b1.y; b_smem[o + 6]  = b1.z; b_smem[o + 7]  = b1.w;
            b_smem[o + 8]  = b2.x; b_smem[o + 9]  = b2.y; b_smem[o + 10] = b2.z; b_smem[o + 11] = b2.w;
            b_smem[o + 12] = b3.x; b_smem[o + 13] = b3.y; b_smem[o + 14] = b3.z; b_smem[o + 15] = b3.w;
        }
        __syncthreads();

        if (x < w4 && y < h4) {
            for (int j = 0; j < MATMUL_DIM; ++j) {
                // a-блок для (localY, j): 16 float
                const float* a_blk = &a_smem[(localY * MATMUL_DIM + j) * SMEM_STRIDE];
                // b-блок для (j, localX): 16 float
                const float* b_blk = &b_smem[(j * MATMUL_DIM + localX) * SMEM_STRIDE];

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        float aik = a_blk[i * 4 + kk];   // broadcast по варпу
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            float bkj = b_blk[kk * 4 + jj]; // broadcast по варпу
                            acc[i * 4 + jj] += aik * bkj;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (x < w4 && y < h4) {
        // Запись результата
        c[(y*4 + 0) * w4 + x] = make_float4(acc[0],  acc[1],  acc[2],  acc[3]);
        c[(y*4 + 1) * w4 + x] = make_float4(acc[4],  acc[5],  acc[6],  acc[7]);
        c[(y*4 + 2) * w4 + x] = make_float4(acc[8],  acc[9],  acc[10], acc[11]);
        c[(y*4 + 3) * w4 + x] = make_float4(acc[12], acc[13], acc[14], acc[15]);
    }
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c,
            unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        (float4*)a.cuptr(), (float4*)b.cuptr(), (float4*)c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda