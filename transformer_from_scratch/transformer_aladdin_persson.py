import torch
import torch.nn as nn
import math

# From : https://www.youtube.com/watch?v=U0s0f995w14

class SelfAttention(nn.Module):

    def __init__(self, embedding_size, no_of_heads):
        """ 
        Embedding size - 256, no of heads - 8 
        Then we will have 32 as each head size
        """
        super(SelfAttention, self).__init__()
        self.embedding_size = embedding_size
        self.no_of_heads = no_of_heads
        self.head_size = embedding_size // no_of_heads

        assert(self.head_size * no_of_heads == embedding_size), "Needs embedding size to be divisible by no of heads"

        # All the 3 k, q, v will have same dimension as the head_size

        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)

        # FC layer dimensions
        self.fc_out = nn.Linear(no_of_heads * self.head_size, embedding_size)

    def forward(self, queries, keys, values, mask):
        # No of examples
        N = queries.shape[0]

        # Shape of keys, values and queries. Basically the length of examples. Query length can be coming 
        # from either encoder or decoder
        keys_len, values_len, queries_len = keys.shape[1], values.shape[1], queries.shape[1] 

        # Reshape all the three for Multi-headed self-attention (embeddings is getting reshaped to no_of_heads*headsize)
        # Eg: 512 = 8 * 64

        values = values.reshape(N, values_len, self.no_of_heads, self.head_size)
        keys = keys.reshape(N, keys_len, self.no_of_heads, self.head_size)
        queries = queries.reshape(N, queries_len, self.no_of_heads, self.head_size)

        # Pass all the three to Linear layers
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Forward pass, Multi-headed self attention
        # This throws errors to deal with batch dimensions
        # attention_output = torch.matmul(nn.softmax(torch.matmul(queries, keys)/math.sqrt(queries.shape(-1))), values)

        # Forward pass, Multi-headed self attention using einsum
        # queries shape: [N, query_len, no_of_heads, head_size]
        # keys shape: [N, keys_len, no_of_heads, head_size]
        # energy shape: [N, no_of_heads, query_len, key_len]
        # In every head, get a score mapping for every element in query len to key length
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Masked Multi-head attention
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))
        
        # Dimension = 3 --> We are doing softmax on the 3rd dimension i.e on the key dimensions
        # So, we can understand which key is important for every query.
        attention_output = torch.softmax(energy/math.sqrt(self.embedding_size), dim=3)

        # attention_output shape: [N, no_of_heads, query_len, key_len]
        # values shape: [N, values_len, no_of_heads, head_size]
        # Output is nothing but for every query we have, we need to get the attention scores 
        # where keys and values dimensions are multiplied
        # output shape: [N, query_len, no_of_heads, head_size]
        output = torch.einsum("nhqk,nvhd->nqhd", [attention_output, values])

        # Concatenate the Multi-headed output again i.e flatten last 2 dimensions
        output = output.reshape(N, queries_len, self.no_of_heads*self.head_size)

        # Fully-Convoluted layer, returns same dimensions
        output = self.fc_out(output)

        return output

class TransformerBlock(nn.Module):

    def __init__(self, embedding_size, no_of_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_size, no_of_heads)

        # Normalizing across the embedding dimension, not the batch dimension!
        self.norm1 = nn.LayerNorm(embedding_size)
        # We have 2 normalisation blocks in architecture
        self.norm2 = nn.LayerNorm(embedding_size)
        
        # Mapping from embedding size -> forward expansion*embedding size -> embedding size
        # eg: 512 -> 1024 -> 512

        # This introduces non-linearity in the entire architecture
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion*embedding_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embedding_size, embedding_size)
        )

        self.dropout = nn.Dropout(dropout)

    
    def forward(self, key, query, value, mask):
        attention = self.attention(query, key, value, mask)

        # "+ query" is the one from skip connection. 
        # This will be input to feed-forward layer (Refer transformer picture)
        x = self.dropout(self.norm1(attention + query))

        # Output from feed-forward is a part of input to the next normalization block
        forward = self.feed_forward(x)

        # The 2nd norm block has input from the feedforward block (forward) and from the previous normalized block i.e x
        output = self.dropout(self.norm2(forward + x))
        return output

class Encoder(nn.Module):
    def __init__(
        self, 
        source_vocab_size, 
        embedding_size,
        num_layers,
        no_of_heads,
        device,
        forward_expansion,
        dropout,
        max_length
    ):
        super(Encoder, self).__init__()

        self.embedding_size = embedding_size
        self.device = device
        self.word_embedding = nn.Embedding(source_vocab_size, embedding_size)
        self.positional_embedding = nn.Embedding(max_length, embedding_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_size,
                    no_of_heads,
                    dropout = dropout,
                    forward_expansion = forward_expansion
                )
            for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # We have N examples with each example having a length of seq_length
        N, seq_length = x.shape

        # arange will have 0, 1, 2, ... seq_length-1
        # expand will simply repeat it N times
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # Add both input embedding and positional embedding and pass it to dropout
        out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))

        for layer in self.layers:
            # Key, query, value are the same in encoder block
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, no_of_heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        # This is used for "Masked Multiheaded self attention"
        self.attention = SelfAttention(embedding_size, no_of_heads)
        self.norm = nn.LayerNorm(embedding_size)

        # This is used for Multi headed cross attention where keys, values come from encoder output
        self.transformer_block = TransformerBlock(embedding_size, no_of_heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, source_mask, target_mask):
        "Target mask is the one used for masked multi headed self attention"
        # Source mask is optional and is used for padding sentences
        # value and key come from encoder.

        # Masked multi-headed self attention (key, query, value are from same source)
        attention = self.attention(x, x, x, target_mask)

        # + x describes the skip connection in Decoder near masked attention block
        # This output acts as an AVERAGE query to look into the encoder keys, values
        query = self.dropout(self.norm(attention + x))

        output = self.transformer_block(key, query, value, source_mask)
        return output

class Decoder(nn.Module):
    def __init__(
        self, 
        target_vocab_size,
        embedding_size,
        num_layers,
        no_of_heads,
        forward_expansion,
        dropout,
        device,
        max_length
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embedding_size)
        self.positional_embedding = nn.Embedding(max_length, embedding_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_size,
                    no_of_heads, 
                    forward_expansion,
                    dropout,
                    device
                )
                for _ in range(num_layers)
            ]
        )

        # Final FC layer which predicts probabilities
        self.fc_output = nn.Linear(embedding_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, source_mask, target_mask):

        N, seq_length = x.shape

        # Create simple positional embeddings which we append to the embedding vector before giving to decoder block
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # Adding both the embedding vector to it's position which serves as input to decoder block
        x = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))

        for layer in self.layers:
            # Cross attention
            x = layer(x, encoder_output, encoder_output, source_mask, target_mask)
        
        output = self.fc_output(x)
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        source_pad_index,
        target_pad_index,
        embedding_size=512,
        num_layers=6,
        forward_expansion=4,
        no_of_heads=8,
        dropout=0,
        device='cuda',
        max_length=100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            source_vocab_size,
            embedding_size,
            num_layers,
            no_of_heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            target_vocab_size,
            embedding_size,
            num_layers,
            no_of_heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.source_pad_index = source_pad_index
        self.target_pad_index = target_pad_index
        self.device = device

    def make_source_mask(self, source):
        # [N, 1, 1, source_length]
        source_mask = (source != self.source_pad_index).unsqueeze(1).unsqueeze(2)
        return source_mask.to(self.device)

    def make_target_mask(self, target):
        N, target_len = target.shape
        # target mask is used in masked multi headed self attention where the output shouldn't see anything after it's 
        # current timestep
        # So, it's basically a lower triangular matrix with 1s
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(
            N, 1, target_len, target_len
        )
        return target_mask.to(self.device)

    def forward(self, source, target):
        source_mask = self.make_source_mask(source)
        target_mask = self.make_target_mask(target)

        # encoder block in transformer
        encode_source = self.encoder(source, source_mask)
        # Decoder block in transformer, see how target_mask is used to prevent seeing future
        output = self.decoder(target, encode_source, source_mask, target_mask)
        return output

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source = torch.tensor([
        [1, 5, 4, 3, 6, 5, 7, 2, 0],
        [1, 8, 3, 5, 6, 7, 5, 2, 0]
    ]).to(device)

    target = torch.tensor([
        [1, 7, 4, 3, 8, 6, 2],
        [1, 9, 3, 7, 6, 8, 2]
    ]).to(device)

    source_pad_index = 0
    target_pad_index = 0
    # Both source and target vocab size are restricted from 0-9. So, anything above 10 will be mapped to 10 in Huggingface
    source_vocab_size = 10
    target_vocab_size = 10

    model = Transformer(source_vocab_size, target_vocab_size, source_pad_index, target_pad_index).to(device)
    # Target shifted to 1 as it needs to predict the end of sentence token
    output = model(source, target[:, :-1])
    print(output.shape)
    print(output)




        