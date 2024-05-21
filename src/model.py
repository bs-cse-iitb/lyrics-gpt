"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import logging
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config """
    
    embd_pdrop = 0.1        # Embedding Layer Dropout
    resid_pdrop = 0.1       # Residual Connection dropout
    attn_pdrop = 0.1        # Attention Dropout

    def __init__(self, vocab_size, block_size, **kwargs):
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        for key, val in kwargs.items():
            setattr(self, key, val)
            
            

class CausalSelfAttention(nn.Module):
    
    """ A Multi Head Attention module """

    def __init__(self, config):
        super().__init__()
        
        assert config.n_embd % config.n_head == 0       # n_embedding must be a multiple of n_head
        self.n_head = config.n_head                     # no of heads
        
        # Key, Query, Value projections for all heads, embedding vector goes into K, Q, V not one-hot 
        self.key   = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # Regularization
        self.attn_drop  = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        
        # output projection : used after concatenation
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        
        
    def forward(self, x):
        
        B, T, C = x.size()              # batch_size x seq_len x channel_dim (embed_dim of token)
        dim_head = C // self.n_head     # Last dimension of each head
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B,   T, self.n_head, dim_head).transpose(1, 2)  # (B, nh, T, dim_head)
        q = self.query(x).view(B, T, self.n_head, dim_head).transpose(1, 2)  # (B, nh, T, dim_head)
        v = self.value(x).view(B, T, self.n_head, dim_head).transpose(1, 2)  # (B, nh, T, dim_head)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) ---> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))      # Q @ K_trans /sqrt(dim_head)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))      # e^{-inf} = 0   
        att = F.softmax(att, dim=-1)                                         # Softmax
        att = self.attn_drop(att)                                            # Atttention dropout
        
        # Softmax(Q @ K_trans /sqrt(dim_head)) @ V  : output of Multi Head Attention Block
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # Output projection : W^{O} in Attetion Paper
        y = self.proj(y)
        y = self.resid_drop(y)
        
        return y


class Block(nn.Module):
    
    """ Block : Multi-Head-Attention-withCrossAttention(x) -----> MLP(x) """

    def __init__(self, config):
        super().__init__()
        
        # Layer Norms
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        # Attention Block
        self.attn = CausalSelfAttention(config)
        
        # Multilayer Perceptron 
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        
        return x
    

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # Input Embeddings and Positional Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)   # Dropout for input embeddings
        
        # Transformer Blocks, this is a decoder only model, using masked attention
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
        # Decoder head : used at the end of the last layer
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)              # Initialise weights recursively

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    
    def get_block_size(self):
        return self.block_size
    
    
    def _init_weights(self, module):
        """
        Recursively initailises the weights of different modules
        """
        
        if isinstance(module, (nn.Linear, nn.Embedding)):
            
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
                
        elif isinstance(module, nn.LayerNorm):
            
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
            

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        """
        Forward Pass of the GPT model
        Args:
            idx (torch.tensor): ids of sentece tokens : shape : batch_size x T
            targets (torch.tensor): right shifted tensor of idx
        """
        
        B, T = idx.size()
        assert T <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)                        # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :T, :]                # each position maps to a (learnable) vector
        
        x = self.drop(token_embeddings + position_embeddings)       # Get initial embeddings ---> Dropout
        x = self.blocks(x)                                          # Pass the embeddings through n_layers blocks sequentially
        x = self.ln_f(x)                                            # Pass the final o/p to Layer Norm
        logits = self.head(x)                                       # Pass this o/p to classifiaction head

        # If targets are given, the loss is also calculated
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
