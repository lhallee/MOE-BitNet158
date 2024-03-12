import torch
import torch.nn as nn
import torch.nn.functional as F
from bitlinear import BitLinear


# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/mixtral/modeling_mixtral.py
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_size = config.hidden_size
        if config.bitnet:
            Linear = BitLinear
        else:
            Linear = nn.Linear
        self.w1 = Linear(self.hidden_size, self.ffn_dim, bias=False)
        self.w2 = Linear(self.ffn_dim, self.hidden_size, bias=False)
        self.w3 = Linear(self.hidden_size, self.ffn_dim, bias=False)

    def forward(self, hidden_states):
        current_hidden_states = F.silu(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states
    

# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/mixtral/modeling_mixtral.py
class TokenTopKMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        h_type = hidden_states.dtype
        bs, sequence_length, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.router(hidden_states.to(self.router.weight.dtype)) # move to gate precision

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (bs * sequence_length, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_size)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(h_type)) # move back to bit
        final_hidden_states = final_hidden_states.reshape(bs, sequence_length, hidden_size)
        return final_hidden_states, router_logits
    

class SentenceTopKMoeBlock(nn.Module):
    def __init__(self, config):
        """
        Sentence level MoE, topk expert aggregated
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.topk = config.num_experts_per_tok
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h_type = hidden_states.dtype
        bs, L, _ = hidden_states.size()
        # Compute router logits
        router_logits = self.router(hidden_states.to(self.router.weight.dtype))  # (bs, L, num_experts)
        router_logits = router_logits.mean(dim=1)  # Average across L dimension (bs, num_experts)
        router_probs = router_logits.softmax(dim=-1)  # (bs, num_experts)

        # Topk
        topk_weights, topk_indices = torch.topk(router_probs, self.topk, dim=-1)  # (bs, topk), (bs, topk)

        # Compute all expert outputs
        expert_outputs = torch.stack([expert(hidden_states) for expert in self.experts], dim=1)  # (bs, num_experts, L, hidden_size)

        # Compute weighted combination of expert outputs
        topk_indices = topk_indices.unsqueeze(1).unsqueeze(-1).expand(-1, L, -1, self.hidden_size)  # (bs, L, topk, hidden_size)
        expert_outputs = expert_outputs.gather(1, topk_indices)  # (bs, L, topk, hidden_size)
        expert_outputs = (expert_outputs * topk_weights.unsqueeze(1).unsqueeze(-1)).sum(dim=2).to(h_type)  # (bs, L, hidden_size)

        return expert_outputs, router_logits  # (bs, L, hidden_size), (bs, num_experts)
    

class VisionMoeBlock(nn.Module):
    def __init__(self, config):
        """
        Sentence level MoE, topk expert aggregated
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.topk = config.num_experts_per_tok
        self.num_channels = config.num_channels
        self.router = nn.Linear(self.hidden_size * self.num_channels, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h_type = hidden_states.dtype
        size = hidden_states.size()
        if len(size) == 3:
            bs, L, d = size
            c = 1
            hidden_states = hidden_states.view(bs, c, L, d)
        else:
            bs, c, L, d = hidden_states.size()

        # Reshape hidden_states to (bs, L, hidden_size * num_channels)
        hidden_states = hidden_states.view(bs, L, -1)

        # Compute router logits
        router_logits = self.router(hidden_states.to(self.router.weight.dtype))  # (bs, L, num_experts)
        router_logits = router_logits.mean(dim=1)  # Average across L dimension (bs, num_experts)
        router_probs = router_logits.softmax(dim=-1)  # (bs, num_experts)

        # Topk
        topk_weights, topk_indices = torch.topk(router_probs, self.topk, dim=-1)  # (bs, topk), (bs, topk)

        # Compute all expert outputs
        expert_outputs = torch.stack([expert(hidden_states.view(bs * L, c, -1)).view(bs, L, -1) for expert in self.experts], dim=1)  # (bs, num_experts, L, hidden_size * num_channels)

        # Compute weighted combination of expert outputs
        topk_indices = topk_indices.unsqueeze(1).unsqueeze(-1).expand(-1, L, -1, self.hidden_size * self.num_channels)  # (bs, L, topk, hidden_size * num_channels)
        expert_outputs = expert_outputs.gather(1, topk_indices)  # (bs, L, topk, hidden_size * num_channels)
        expert_outputs = (expert_outputs * topk_weights.unsqueeze(1).unsqueeze(-1)).sum(dim=2).to(h_type)  # (bs, L, hidden_size * num_channels)

        # Reshape expert_outputs back to (bs, c, L, hidden_size)
        expert_outputs = expert_outputs.view(bs, c, L, -1)

        return expert_outputs, router_logits  # (bs, c, L, hidden_size), (bs, num_experts)