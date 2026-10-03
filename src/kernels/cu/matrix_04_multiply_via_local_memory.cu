#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#include <cublas_v2.h>

#define HD __host__ __device__

struct float2x2 {
  HD float2x2(float2 row1, float2 row2) {
    col[0] = { row1.x, row2.x };
    col[1] = { row1.y, row2.y };
  }
  float2x2() = default;
  float2 col[2];
};

HD void fma2x2(const float2x2 &a, const float2x2 &b, float2x2 &acc) {
  acc.col[0].x += a.col[0].x * b.col[0].x;
  acc.col[0].x += a.col[1].x * b.col[0].y;
  acc.col[0].y += a.col[0].y * b.col[0].x;
  acc.col[0].y += a.col[1].y * b.col[0].y;
  acc.col[1].x += a.col[0].x * b.col[1].x;
  acc.col[1].x += a.col[1].x * b.col[1].y;
  acc.col[1].y += a.col[0].y * b.col[1].x;
  acc.col[1].y += a.col[1].y * b.col[1].y;
}

__global__ void matrix_multiply_via_local_memory(
                       const float2* a, // rows=h x cols=k
                       const float2* b, // rows=k x cols=w
                             float2* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ float2x2 alocal[MATMUL_DIM][MATMUL_DIM];
    __shared__ float2x2 blocal[MATMUL_DIM][MATMUL_DIM];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t localX = threadIdx.x;
    uint32_t localY = threadIdx.y;
    float2x2 res = {};

    uint32_t w2 = w/2;
    uint32_t h2 = h/2;
    uint32_t k2 = k/2;

    for (int bi = 0; bi < k2/MATMUL_DIM; ++bi) {
      if (x < w2 && y < h2) {
        alocal[localY][localX] = { a[y*2 * k2 + bi * MATMUL_DIM + localX], a[(y*2+1) * k2 + bi * MATMUL_DIM + localX] };
        blocal[localY][localX] = { b[(bi * MATMUL_DIM + localY) * 2 * w2 + x], b[((bi * MATMUL_DIM + localY)*2 + 1) * w2 + x] };
      }
      __syncthreads();
      if (x < w2 && y < h2) {
        for (int j = 0; j < MATMUL_DIM; ++j) {
          fma2x2(alocal[localY][j], blocal[j][localX], res);
        }
      }
      __syncthreads();
    }

    if (x < w2 && y < h2) {
      c[y * 2 * w2 + x] = { res.col[0].x, res.col[1].x };
      c[(y*2+1) * w2 + x] = { res.col[0].y, res.col[1].y }; 
    }
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    // Создаём handle cuBLAS (можно сделать статическим, чтобы не создавать каждый раз)
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // ВАЖНО: cuBLAS Column-Major.
    // Мы хотим C = A * B (Row-Major).
    // Это эквивалентно C^T = B^T * A^T (Column-Major).
    // Поэтому:
    //   - "m" и "n" в cuBLAS — это размеры C^T.
    //   - C (h x w) => C^T (w x h) => m = w, n = h.
    //   - "k" — общая размерность (глубина) = k.
    //   - "A" в cuBLAS — это B (k x w Row-Major) => в Column-Major это (w x k).
    //   - "B" в cuBLAS — это A (h x k Row-Major) => в Column-Major это (k x h).
    
    // Аргументы cublasSgemm:
    // cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    
    cublasSgemm(handle,
                CUBLAS_OP_N,  // Не транспонируем "B" (в нашем случае это матрица A)
                CUBLAS_OP_N,  // Не транспонируем "A" (в нашем случае это матрица B)
                w,            // m: строки в C^T (столбцы в C, т.е. w)
                h,            // n: столбцы в C^T (строки в C, т.е. h)
                k,            // k: глубина
                &alpha,
                (const float*)b.cuptr(), w,  // "A" = B (k x w), lda = w (ширина B в Row-Major)
                (const float*)a.cuptr(), k,  // "B" = A (h x k), ldb = k (ширина A в Row-Major)
                &beta,
                (float*)c.cuptr(), w);       // "C" = C (h x w), ldc = w (ширина C в Row-Major)

    cublasDestroy(handle); // Если handle не статический
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
