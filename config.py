from transformers.configuration_utils import PretrainedConfig


class BitformerConfig(PretrainedConfig):
    model_type = 'bitformer'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=8,
        hidden_act='silu',
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=1e6,
        sliding_window=4096,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        output_router_logits=True,
        router_aux_loss_coef=0.001,
        is_causal=False, #GPT or BERT
        moe=True, # MOE or not
        bitnet=True, #bitlinear or not
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        self.is_causal = is_causal
        self.moe = moe
        self.bitnet = bitnet

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )