#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void mean_filter_kernel(unsigned char* output, unsigned char* input, int width, int height, int radius){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = threadIdx.z; 

    // depending on whether channel is 0, 1, 2 -> we will be focusing on either r, g, or b!
    int baseOffset = channel * height * width;
    if (col < width && row < height){
        int pixValues = 0;
        int pixels = 0;
        
        // iterate from -radius till radius and check for condition if it is out of bounds!
        for (int blurRow= -radius; blurRow <= radius; blurRow+=1){
            for (int blurCol= -radium; blurCol <= radius; blurCol+= 1){
                int currRow = row + blurRow;
                int currCol = col + blurCol;

                // which means the radius is in the bounds of image!
                if (currRow >= 0 && currRow < height && currCol >= 0 && currCol < width){
                 pixVal += input[baseOffset + ((currRow * width) + currCol)];
                 pixels += 1;
                }
            }
        }
        // average out the pixel values according to how many pixels!
        output[baseOffset + row * width + col] = (unsigned char)(pixVal / pixels);
    }
}

// helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

torch::Tensor mean_filter(torch::Tensor image, int radius){
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);
    assert(radius > 0);

    const auto channels = image.size(0);
    const auto height = image.size(1);
    const auto width = image.size(2);

    auto result = torch::empty_like(image);

    // channels is 3, that we have given indirectly like this in z axis!
    dim3 threads_per_block(16, 16, channels);
    dim3 number_of_blocks(
        cdiv(width, threads_per_block.x),
        cdiv(height, threads_per_block.y)
    )

    mean_filter_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height,
        radius
    )
    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}