import numpy as np
import timm
from vit_architecture import VisionTransformer
import torch

# Helper function
def get_n_params(module):
    # This gives the total learnable parameters
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()
    np.testing.assert_allclose(a1, a2)

model_name = 'vit_base_patch16_384'
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()
print(type(model_official))

custom_config = {
    "image_size": 384,
    "patch_size": 16,
    "input_channels": 3,
    "embed_dim": 768,
    "depth": 12,
    "n_heads": 12,
    "mlp_ratio": 4,
    "qkv_bias":True
}

model_custom = VisionTransformer(**custom_config)
model_custom.eval()
print(type(model_custom))

# Checking the original (o) and custom (c) networks has same number of paramters
# This check passes only if the order in which the custom model components are instantiated
# are same as that of original model

for (n_o, p_o), (n_c, p_c) in zip(
    model_official.named_parameters(), model_custom.named_parameters()
    ):
    # Making sure that no of parameters in each layer is the same 
    assert p_o.numel() == p_c.numel()
    print(f"{n_o} | {n_c}")

    # All the parameters in the custom module are replaced by the original parameters
    p_c.data[:] = p_o.data

    # Checking that the paramters are copied properly
    assert_tensors_equal(p_c.data, p_o.data)

# Random tensor
inp = torch.randn(1, 3, 384, 384)
res_c = model_custom(inp)
res_o = model_official(inp)

# Checking that both the models has same total number of parameters 
assert get_n_params(model_custom) == get_n_params(model_official)
assert_tensors_equal(res_o, res_c)




