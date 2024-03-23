#include <cuda.h>
#include <stdio.h>

// this does element-wise addition in the kernel
// kernels doesn't return anything
__global__ void vectorAddKernel(float* A, float* B, float* C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // check for boundary conditions
    if (i < n){
        C[i] = A[i] + B[i];
    }
}

// have a function/macro that catches the error and prints in an understandable way
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

// macro for ceiling division
inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1)/b;
}

void vecAdd(float *A, float *B, float *C, int n){
    // declaring new variables which will store the dim of these vectors
    float *A_d, *B_d, *C_d;
    size_t size = n * sizeof(float);

    // Allocate the memory for these pointers in device
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    const unsigned int numThreads = 256;
    unsigned int numBlocks = cdiv(n, numThreads);

    // calling the kernel
    vectorAddKernel<<<numBlocks, numThreads>>>(A_d, B_d, C_d, n);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // move the elements from device to host (C_d is device, C is host)
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    // free elements in device
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
  const int n = 1000;
  float A[n];
  float B[n];
  float C[n];

  // generate some dummy vectors to add
  for (int i = 0; i < n; i += 1) {
    A[i] = float(i);
    B[i] = A[i] / 1000.0f;
  }

  vecAdd(A, B, C, n);

  // print result
  for (int i = 0; i < n; i += 1) {
    if (i > 0) {
      printf(", ");
      if (i % 10 == 0) {
        printf("\n");
      }
    }
    printf("%8.3f", C[i]);
  }
  printf("\n");
  return 0;
}
