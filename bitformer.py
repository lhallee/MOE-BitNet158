import torch
import torch.nn as nn

from typing import Optional, Tuple
from transformers.models.mixtral.modeling_mixtral import MixtralModel, MixtralForCausalLM

from attention import SelfAttention, SelfFlashAttention
from moe_blocks import SentenceTopKMoeBlock, TokenTopKMoeBlock, MLP


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MixtralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class BitformerLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        if config._attn_implementation == 'flash_attention_2':
            self.self_attn = SelfFlashAttention(config, layer_idx=layer_idx)
        else:
            self.self_attn = SelfAttention(config, layer_idx=layer_idx)
        self.moe = config.moe
        if config.moe:
            if config.is_causal:
                self.MLP = TokenTopKMoeBlock(config)
            else:
                self.MLP = SentenceTopKMoeBlock(config)
        else:
            self.MLP = MLP
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.moe:
            hidden_states, router_logits = self.MLP(hidden_states)
        else:
            hidden_states = self.MLP(hidden_states)
            router_logits = None
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs
    

class BitformerModel(MixtralModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([BitformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        self._attn_implementation = config._attn_implementation


class BitformerForLM(MixtralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = BitformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()


class VisionBitformer(MixtralModel): # TODO
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([BitformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        self._attn_implementation = config._attn_implementation