# NOTE: Some version error for 3.12 while using torch.compile()!!!
# Example 1 - Using TORCH_LOGS to get Triton kernel for Pytorch code!
# Call using TORCH_LOGS="output_code" python compile_square.py
import torch
a = torch.randn(4)
b = torch.compile(torch.square(a))
# b = torch.square(a)
print(a)
print(b)

## Example 2 - Understanding Fusion
import torch
def square(a):
    # In Eager mode (i.e without torch.compile)
    # Read A, write back to A, read A and return A. Pytorch fires up 2 kernels to do this
    a = torch.square(a)
    return torch.square(a)

# When using compile, this fuses both squares, so it will do Read A, do the square (but wont write it to global memory), use it to do square again and returns the final value
optimized_square = torch.compile(square)
optimized_square(torch.randn(1000, 1000).cuda())
