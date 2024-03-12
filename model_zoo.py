import torch
import torch.nn as nn

from typing import Optional, Tuple, Union
from transformers import (
    BertForSequenceClassification,
    MixtralModel,
    MixtralForCausalLM,
    MixtralForSequenceClassification,
    PreTrainedModel
)
from transformers.modeling_outputs import MoeModelOutputWithPast, SequenceClassifierOutput, SemanticSegmenterOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from attention import SelfAttention, SelfFlashAttention, VisionAttention
from moe_blocks import SentenceTopKMoeBlock, TokenTopKMoeBlock, VisionMoeBlock, MLP


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
        if config.attention_type == 'flash_attention_2':
            config._attn_implementation = 'flash_attention_2'
            self.self_attn = SelfFlashAttention(config, layer_idx=layer_idx)
        elif config.attention_type == 'vision':
            config._attn_implementation = 'sdpa'
            self.self_attn = VisionAttention(config)
        else:
            config._attn_implementation = 'sdpa'
            self.self_attn = SelfAttention(config, layer_idx=layer_idx)
        self.hidden_size = config.hidden_size
        self.moe = config.moe
        if config.moe:
            if config.is_causal:
                self.MLP = TokenTopKMoeBlock(config)
            elif config.attention_type == 'vision':
                self.MLP = VisionMoeBlock(config)
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
        self.post_init()


class BERTSequenceClassifier(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BitformerModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()


class GPTSequenceClassifier(MixtralForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = BitformerModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()


class BitformerForSequenceClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__()
        if config.is_causal:
            self.model = GPTSequenceClassifier(config)
        else:
            self.model = BERTSequenceClassifier(config)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class VisionBitformer(MixtralModel):
    # No tokenization, requires that hidden_size is image size and consistent
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([BitformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.config = config
        self.gradient_checkpointing = False
        self.post_init()
        self._attn_implementation = config._attn_implementation

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        seq_length = inputs_embeds.shape[-2]
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)


            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            next_cache = None
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits] if v is not None)
            
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )
    

class VisionBitformerForImageClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = VisionBitformer(config)
        self.num_labels = config.num_labels
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        size = hidden_states.size()
        bs, d = size[0], size[-1]
        hidden_states = hidden_states.view(bs, -1, d)
        pooled_output = hidden_states.max(dim=1).values # max pooling
        logits = self.score(pooled_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    

class VisionBitformerForSemanticSegmentation(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = VisionBitformer(config)
        self.num_labels = config.num_labels
        self.kernel_size = config.kernel_size
        self.stride = 1
        self.padding = (self.kernel_size - 1) // 2
        self.conv_seg = nn.Conv2d(
            config.num_channels,
            self.num_labels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        self.post_init()

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0] # (bs, c, h, w)
        print(hidden_states.shape)
        logits = self.conv_seg(hidden_states)
        print(logits.shape)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )