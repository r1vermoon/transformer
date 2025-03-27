from torch import nn

class TokenEmbedding(nn.Module):
    
    def __init__(self, vocab_size,d_model):
        super().__init__(vocab_size,d_model,padding_idx=1)