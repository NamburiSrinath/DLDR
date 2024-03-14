# This creates a cpp modules! So, we can use Python and Pytorch libs to create cpp versions
import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
std::string hello_world() {
  return "Hello World!";
}
"""

my_module = load_inline(
    name='my_module',
    cpp_sources=[cpp_source],
    functions=['hello_world'],
    verbose=True,
    build_directory='./tmp'
)

print(my_module.hello_world())