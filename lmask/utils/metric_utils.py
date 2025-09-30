import torch


def compute_valid_average(src):
    valid = src[torch.isfinite(src)]
    avg = valid.mean().item() if len(valid) > 0 else float("nan")
    return avg


def compute_reward_and_gap_averages(masked_reward, cost_bks):
    valid_avg = compute_valid_average(masked_reward)
    gap = (-masked_reward - cost_bks) / cost_bks
    valid_gap = compute_valid_average(gap)
    return valid_avg, valid_gap


def get_filtered_max(reward, feasible_mask):
    """Get mean of maximum feasible rewards.
    Args:
        reward: tensor containing rewards
        feasible_mask: boolean mask for feasible solutions
    Returns:
        float: mean of maximum feasible rewards, or -inf if no feasible solutions
    """
    masked_reward = reward.masked_fill(~feasible_mask, float("-inf"))
    for dim in range(masked_reward.dim() - 1, 0, -1):
        masked_reward = masked_reward.max(dim=dim)[0]
    filtered = masked_reward[masked_reward > float("-inf")]
    return torch.tensor(float("-inf"), device=reward.device) if len(filtered) == 0 else filtered.mean()


def count_unique_permutations(actions, sol_feas=None, return_unique_sols=False):
    """
    Args:
        actions: Tensor of shape [B, S, n] containing permutations
        sol_feas: Optional tensor of shape [B, S] indicating feasible solutions
        return_unique_sols: Boolean, whether to return unique solutions
    Returns:
        counts: Tensor of shape [B] containing counts of unique feasible permutations
        unique_sols: List of lists containing unique feasible permutations (if return_unique_sols=True)
    """
    B, S, n = actions.size()
    counts = []
    all_unique_sols = []

    for b in range(B):
        if sol_feas is not None:
            # Filter feasible solutions
            feasible_perms = actions[b][sol_feas[b]]
            batch_perms = [tuple(perm.tolist()) for perm in feasible_perms]
        else:
            batch_perms = [tuple(perm.tolist()) for perm in actions[b]]

        unique_perms = set(batch_perms)
        counts.append(len(unique_perms))

        if return_unique_sols:
            unique_sols = [list(perm) for perm in unique_perms]
            all_unique_sols.append(unique_sols)

    counts = torch.tensor(counts, device=actions.device)

    if return_unique_sols:
        return counts, all_unique_sols
    return counts


def calculate_sampling_metrics(out):
    out = out.reshape(out.batch_size[0], -1)
    actions, sol_feas, reward = out["actions"], out["sol_feas"], out["reward"]
    ins_feas = out["sol_feas"].any(-1)
    max_aug_reward = get_filtered_max(reward, sol_feas)
    num_unique_feas_sols = count_unique_permutations(actions, sol_feas)

    return {
        "max_aug_reward": round(max_aug_reward.item(), 2),
        "ins_feas": ins_feas.float().mean().item(),
        "num_unique_feas_sols": num_unique_feas_sols.float().mean().item(),
    }


# Function to calculate service start times and wait times for a given route
def calculate_schedule_metrics(td, route: torch.Tensor):
    duration_matrix = td["duration_matrix"]
    time_windows = td["time_windows"]
    batch_size, seq_len = route.size()
    device = route.device

    service_start_times = torch.zeros((batch_size, seq_len), device=device)
    wait_times = torch.zeros((batch_size, seq_len), device=device)

    earliest_times = time_windows[..., 0]
    batch_idx = torch.arange(batch_size, device=device)[:, None]

    for t in range(1, seq_len):
        prev_node = route[:, t - 1]
        curr_node = route[:, t]

        arrival_time = service_start_times[:, t - 1] + duration_matrix[batch_idx[:, 0], prev_node, curr_node]

        wait_times[:, t] = torch.clamp(earliest_times[batch_idx[:, 0], curr_node] - arrival_time, min=0.0)
        service_start_times[:, t] = arrival_time + wait_times[:, t]

    return service_start_times, wait_times
