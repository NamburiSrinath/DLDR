#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void rgb_to_grayscaleKernel(unsigned char* output, unsigned char* input, int width, int height){
    const int channels = 3;

    // Read more about multi-dimensional indexing for blocks/threads
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // checking for boundary conditions
    if (col < width && row < height){
        // the entire image can be thought of flattening index. 
        // So, access the right index to replace the rgb ones
        int outputOffset = row * width + col;
        // the actual inputoffset varies depending on how our input is formatted
        int inputOffset = (row * width + col) * channels;

        // accessing rgb images! this is one way to access if we flatten in a particular way 
        // in rgb_grayscale.py! 
        unsigned char r = input[inputOffset + 0];   // red
        unsigned char g = input[inputOffset + 1];   // green
        unsigned char b = input[inputOffset + 2];   // blue

        output[outputOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

// helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

torch::Tensor rgb_to_grayscale(torch::Tensor image){
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);

    const auto height = image.size(0);
    const auto width = image.size(1);

    auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(image.device()));

    dim3 threads_per_block(16, 16); // use 256 threads per block
    // width/x will give number of blocks needed! think x as x axis
    dim3 num_of_blocks((cdiv(width, threads_per_block.x), cdiv(height, threads_per_block.y)));

    rgb_to_grayscaleKernel<<<num_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height
    )

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}