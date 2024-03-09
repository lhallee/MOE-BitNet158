import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers.models.mixtral.modeling_mixtral import MixtralAttention, MixtralFlashAttention2
from transformers.utils import is_flash_attn_greater_or_equal_2_10

from embeddings import RotaryEmbedding
from bitlinear import BitLinear


class SelfAttention(MixtralAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers" and 
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = config.is_causal
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        if config.bitnet:
            Linear = BitLinear
        else:
            Linear = nn.Linear
        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )


class SelfFlashAttention(MixtralFlashAttention2):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers" and 
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = config.is_causal
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        if config.bitnet:
            Linear = BitLinear
        else:
            Linear = nn.Linear
        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        


#Other version
"""# Adapted from https://github.com/nindanaoto/nanoGPT-BitNet158b/blob/master/model.py
class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.causal = config.causal
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, hidden_states, attention_mask=None):
        bs, L, d = hidden_states.size() # bs, L, d

        q, k, v  = self.c_attn(hidden_states).split(self.n_embd, dim=2)
        k = k.view(bs, L, self.n_head, d // self.n_head).transpose(1, 2) # (bs, nh, L, hs)
        q = q.view(bs, L, self.n_head, d // self.n_head).transpose(1, 2) # (bs, nh, L, hs)
        v = v.view(bs, L, self.n_head, d // self.n_head).transpose(1, 2) # (bs, nh, L, hs)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=self.causal)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attention_mask is not None:
                att = att.masked_fill(attention_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)

        y = y.transpose(1, 2).contiguous().view(bs, L, d)
        y = self.c_proj(y)
        return y"""