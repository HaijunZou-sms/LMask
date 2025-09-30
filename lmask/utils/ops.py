import torch
from torch import Tensor
import torch.nn.functional as F
from tensordict import TensorDict
from rl4co.utils.ops import gather_by_index
from rl4co.models.zoo.am.decoder import PrecomputedCache


def get_action_from_logits_and_mask(logits, mask, tan_clipping=10.0, decode_type="sampling", return_logprobs=False):
    logits = torch.tanh(logits) * tan_clipping
    logits[~mask] = float("-inf")
    logprobs = F.log_softmax(logits, dim=-1)  # [B, n+1]
    probs = logprobs.exp()
    if decode_type == "sampling":
        action = probs.multinomial(1).squeeze(-1)
    elif decode_type == "greedy":
        action = probs.argmax(-1)

    return (logprobs, action) if return_logprobs else action


def slice_cache(hidden: PrecomputedCache, step_forward_idx: Tensor) -> PrecomputedCache:
    """Slice the precomputed cache for active instances"""
    cache_slice = []
    for emb in hidden.fields:
        if isinstance(emb, Tensor) or isinstance(emb, TensorDict):
            cache_slice.append(emb[step_forward_idx])
        else:
            cache_slice.append(emb)

    return PrecomputedCache(*cache_slice)


def get_time_window_violations(td, actions: Tensor) -> Tensor:
    """
    Optimized vectorized implementation, eliminating explicit loops
    Args:
        td: TensorDict containing:
            - time_windows: [B, n+1, 2]
            - duration_matrix: [B, n+1, n+1]
        actions: [B, n] customer node permutation
    Returns:
        violations: [B, n+1] time window violation degree for each node
    """
    batch_size, n = actions.shape
    device = actions.device

    paths = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long, device=device), actions], dim=1)

    time_windows = td["time_windows"].gather(1, paths.unsqueeze(-1).expand(-1, -1, 2))
    tw_early, tw_late = time_windows.unbind(-1)

    batch_idx = torch.arange(batch_size, device=device)[:, None]
    durations = td["duration_matrix"][batch_idx, paths[:, :-1], paths[:, 1:]]

    service_start = torch.empty(batch_size, n + 1, device=device)
    service_start[:, 0] = torch.maximum(torch.zeros(batch_size, device=device), tw_early[:, 0])

    for step in range(1, n + 1):
        arrival_time = service_start[:, step - 1] + durations[:, step - 1]
        service_start[:, step] = torch.maximum(arrival_time, tw_early[:, step])

    time_window_violation = torch.clamp_min(service_start - tw_late, 0)
    return time_window_violation


def get_tour_length(td, actions: Tensor) -> Tensor:
    ordered_locs = torch.cat([td["locs"][:, 0:1], gather_by_index(td["locs"], actions)], dim=1)
    diff = ordered_locs - ordered_locs.roll(-1, dims=-2)
    tour_length = diff.norm(dim=-1).sum(-1)
    return tour_length
