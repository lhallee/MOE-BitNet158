import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers.models.mixtral.modeling_mixtral import MixtralAttention, MixtralFlashAttention2
from transformers.utils import is_flash_attn_greater_or_equal_2_10

from embeddings import RotaryEmbedding
from bitlinear import BitLinear


class SelfAttention(MixtralAttention):

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
        

class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.attention_size % config.num_attention_heads == 0
        if config.bitnet:
            Linear = BitLinear
        else:
            Linear = nn.Linear
        self.qkv = Linear(config.num_channels * config.hidden_size,
                          3 * config.attention_size * config.num_channels,
                          bias=False)
        self.o = Linear(config.num_channels * config.attention_size,
                        config.hidden_size * config.num_channels,
                        bias=False)
        self.n_head = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_size = config.attention_size
        self.dropout = config.attention_dropout
        self.is_causal = config.is_causal
        #self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, hidden_states, *args, **kwargs):
        size = hidden_states.size()
        if len(size) == 3:
            bs, L, d = size
            c = 1
            hidden_states = hidden_states.view(bs, c, L, d)
        else:
            bs, c, L, d = hidden_states.size()
        
        # Reshape hidden_states to (bs * c, L, d)
        hidden_states = hidden_states.view(bs, L, d * c)
        
        q, k, v = self.qkv(hidden_states).split(self.attention_size * c, dim=-1) # (bs, L, hs * c)
        k = k.view(bs, L, self.n_head, (self.attention_size * c) // self.n_head).transpose(1, 2) # (bs, nh, L, hs * c)
        q = q.view(bs, L, self.n_head, (self.attention_size * c) // self.n_head).transpose(1, 2) # (bs, nh, L, hs * c)
        v = v.view(bs, L, self.n_head, (self.attention_size * c) // self.n_head).transpose(1, 2) # (bs, nh, L, hs * c)

        #y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=self.is_causal)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v # (bs, nh, L, L) x (bs, nh, L, hs * c) -> (bs, nh, L, hs * c)

        y = y.transpose(1, 2).contiguous().view(bs, L, self.attention_size * c)
        y = self.o(y)
        
        # Reshape y back to (bs, c, L, d)
        y = y.view(bs, c, L, d)
        
        return y, att, None