"""
Inspired from : https://www.youtube.com/watch?v=ovB0ddFtzzA
"""

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    We try to split an image into patches and embed them
    image_size -> size of the image
    patch_size -> size of the patch
    input channels -> usually 3
    embedding_dimension

    --- Attributers---
    n_patches -> no of patches inside the image
    projection -> nn.Conv2D -> Used to split the images to patches in a easy way
    kernel size and stride are same as patch size which means we get non-overlapping patches
    """

    def __init__(self, image_size, patch_size, input_channels = 3, embedding_dimension=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2

        # Creating non-overlapping patches by making the kernel size and stride equal to patch size
        self.proj = nn.Conv2d(
            input_channels,
            embedding_dimension,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        """
        Input -> n_samples, n_channels, input_height, input_width
        Output -> n_samples, n_patches, embedding_dimension

        Each sample (image) has n_patches and each patch is of embedding dimension length
        """

        x = self.proj(
            x
        )
        # This outputs n_samples, embedding_dimension (as the output channels), n_patches**0.5, n_patches**0.5
        # The height and width of the output can be calculated by treating it like a simple CNN formula

        # The last 2 dimensions are flattened i.e just multiply in dimension space
        x = x.flatten(2) # n_samples, embedding_dimension, n_patches
        x = x.transpose(1, 2) # n_samples, n_patches, embedding_dimension
        return x
    
class Attention(nn.Module):
    """
    dim -> The input and output dimension of every token feature
    n_heads -> No of attention heads
    qkv_bias -> Tells if we need to include bias when projecting Query, Key and Value values
    attention_p -> Dropout probability applied ot key, query and value [Needed during traiing]
    projection_p -> Dropout probability applied to tensor [Needed during training]

    model = torch.nn.Dropout(p)
    model.training -> True ==> It's set when in training mode and does Dropout
    When we do model.eval(), the dropout layer behaves as an Identity mapping

    --- Attributes ---
    scale -> to scale down the dot product in attention i.e normalizing
    qkv -> Linear projection (W) for query, key and value
    proj -> Linear mapping from concatenated output of key, query, value to a new space
    attention dropout, projection droput -> dropouts for both attention and projection heads
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attention_p=0.0, projection_p=0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        # The entire dimension will be split among n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5 # (the normalization denominator)

        # Can create q, k, v linear projections separately, but not needed
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)

        self.attention_dropout = nn.Dropout(attention_p)
        # Projection layer which basically concatenates and returns the same as input dimension
        self.projection = nn.Linear(dim, dim)

        self.projection_dropout = nn.Dropout(projection_p)

    
    def forward(self, x):
        """
        Input and output has the same shape
        n_samples, n_patches +  1 (This is for CLS token), dimension
        """

        n_samples, n_tokens, dim = x.shape

        # Make sure that the input dimension is same as the dimension given to us in constructor 
        # i.e (self.dim)
        if dim != self.dim:
            return ValueError

        # Linear --> The last dimension of input tensor == input dimension of nn.Linear
        # and that gets mapped to output dimension of nn.Linear
        qkv = self.qkv(x) # n_samples, n_patches + 1, dim*3
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )

        qkv = qkv.permute(
            2, 0, 3, 1, 4
        ) # (3, n_samples, n_heads, n_tokens (which is n_patches+1), head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Transposing the keys to compute the dot product
        k_t = k.transpose(-2, -1) #n_samples, n_heads, head_dim, n_tokens

        # Computing dot product between query and key and scaling it
        dp = (
            q @ k_t
        ) * self.scale #n_samples, n_heads, n_tokens, n_tokens

        # Softmax -> Sum will be 1, so we are sort of getting coefficients
        attention = dp.softmax(dim=-1) # n_samples, n_heads, n_tokens, n_tokens
        # Attention dropout layer
        attention = self.attention_dropout(attention)

        # Just multiply the softmax values (alphas) with the values which will be the weighted average attention vector
        weighted_average = attention @ v #n_samples, n_heads, n_tokens, head_dim
        weighted_average = weighted_average.transpose(
            1,2
        ) #n_sample, n_tokens (n_patches+1), n_heads, head_dim

        weighted_average = weighted_average.flatten(2) # n_samples, n_tokens, n_heads*head_dim (which will be dim)

        # Projecting to a linear layer which returns the same dimension
        x = self.projection(weighted_average)
        # Projection dropout before returning final values
        x = self.projection_dropout(x)
        return x

class MLP(nn.Module):
    """
    input_features, hidden_features, output_features -> tells no of nodes at each layer
    p -> Dropout probability

    Attributes:
    fc -> nn.Linear layer, act -> nn.GELU (activation function)
    nn2 -> hidden layer, drop -> dropout layer
    """
    def __init__(self, input_features, hidden_features, output_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, output_features)
        self.drop = nn.Dropout(p)
    
    def forward(self, x):
        # x -> n_samples, n_tokens (n_patches + 1), input_features
        x = self.fc1(x) # n_samples, n_tokens, hidden_features
        x = self.act(x) # n_samples, n_tokens, hidden_features
        x = self.drop(x) # n_samples, n_tokens, hidden_features
        x = self.fc2(x) # n_samples, n_tokens, output_features
        x = self.drop(x) # n_samples, n_tokens, output_features
        return x

class Block(nn.Module):
    """
    dim -> Embedding dimension (usually 768)
    n_heads -> Number of attention heads
    mlp_ratio -> Determines the hidden layer size in the MLP with respect to dim
    qkv_bias -> If True, we add a bias term to Query, Key and Value weights
    p, attention_p -> Dropout and attention dropout probabilities

    ---Attributes ---
    norm1, norm2 -> 2 Normalization layers
    attention -> 1 attention module
    mlp -> 1 MLP block
    """
    def __init__(self, dim, n_heads, mlp_ratio=4, qkv_bias=True, p=0., attention_p=0.):
        super().__init__()
        # eps=1e-6 is a standard to make sure we won't get infinity in case variance is 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        # This is basically the attention block
        self.attention = Attention(
            dim, 
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attention_p=attention_p,
            projection_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        # Creating a bottleneck layer in MLP
        hidden_features = int(mlp_ratio * dim) 
        self.mlp = MLP(
            input_features = dim,
            hidden_features = hidden_features,
            output_features = dim,
            p = p
        )
    
    def forward(self, x):
        # Shape of x doesn't change... n_samples, n_tokens, dim
        # Here we are basically doing the residual connection i.e passing input as well
        x = x + self.attention(self.norm1(x)) 
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """
    img_size -> Given a square image
    patch_size -> We usually deal with square patches
    input_channels -> no of channels, usually 3
    num_classes -> Number of classes for classification task
    embed_dim -> Dimensionality of token/patch that will be the input to transformer
    depth -> No of transformer blocks

    n_heads -> No of attention heads in each attention layer 
    i.e we can pay attention to multiple components of an image in one layer itself

    mlp_ratio -> Determines the hidden dimension of the MLP module
    qkv_bias -> Bias terms for query, key and value projections
    attention_p -> Dropout probability of attention module
    projection_p -> Dropout probability in MLP and other Linear projections inside the attention block


    -----ATTRIBUTES-----
    patch_embed -> Instance of Patch Embedding class

    cls_token -> nn.Parameter (this is the one which gives the probability of each class in our dataset)
    It has embed_dim no of elements (This is basically the answer we are looking for)

    pos_embedding -> Positional embedding for every patch + cls token
    It has (n_patches + 1) * embed_dim elements

    pos_dropout-> nn.Dropout
    blocks -> List of Block modules (nn.ModuleList)
    norm -> Layer normalization

    """
    def __init__(self, 
                image_size=384,
                patch_size=16,
                input_channels=3,
                num_classes=1000,
                embed_dim=768,
                depth=12,
                n_heads=12,
                mlp_ratio=4.0,
                qkv_bias=True,
                attention_p=0.0,
                projection_p=0.0
    ):
        super().__init__()

        # First we create a patch_embed object from the class
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            input_channels=input_channels,
            embedding_dimension=embed_dim
        )

        # Class token parameter initialized to 0s
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Postional embedding for every token (n_patches + 1) whose dimension is same as 768 i.e embed_dimension
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )

        self.pos_dropout = nn.Dropout(projection_p)
        # Creating the whole Transformer encoder structure
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attention_p=attention_p,
                    p = projection_p
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # Takes the final embedding of the CLS token and projects it to num_classes dimensions which is basically the 
        # logits for classification 
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Input dimensions: x -> n_samples, input_channels, img_size, img_size
        Output dimension: Logits over the classes -> n_samples, n_classes
        """
        n_samples = x.shape[0]
        # Convert batches of images to batches of patches (image to patches basically!!) 
        x = self.patch_embed(x) #n_samples, n_patches, embed_dim

        # Replicate the learnable CLS token across the num_samples.
        # n_samples, 1, embed_dim
        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        )

        # Concatenate along the 1st dimension
        x = torch.cat((cls_token, x), dim=1) #n_samples, n_patches + 1, embedding_dimension

        # Add the positional embedding to the original inputs
        x = x + self.pos_embedding # n_samples, n_patches + 1, embedding_dimension
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)

        cls_token = x[:, 0] # Select the class embedding for all samples
        # n_samples, 1, embed_dimension
        
        # Only the final dimension is changed when we deal with nn.Linear
        x = self.head(cls_token) # n_samples, 1, num_classes 

        return x




        

















