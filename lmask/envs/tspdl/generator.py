import torch
import numpy as np
from tensordict.tensordict import TensorDict
from rl4co.envs.common.utils import Generator


class TSPDLGenerator(Generator):
    def __init__(self, num_loc: int = 49, hardness: str = "hard", draft_method: str = "rejection", normalize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.hardness = hardness
        self.normalize = normalize
        self.draft_method = draft_method

    def _generate(self, batch_size):
        # Coordinates: depot (0) + ports (1~num_loc)
        locs = torch.rand((*batch_size, self.num_loc + 1, 2))

        # demand: 0 for depot, 1 for ports
        demand = torch.cat(
            [torch.zeros((*batch_size, 1)), torch.ones((*batch_size, self.num_loc))],
            dim=-1,
        )

        # Draft limit generation
        total_load = self.num_loc  # Sum of port demand
        if self.draft_method == "rejection":
            draft_limit = self._generate_draft_limit_by_rejection(batch_size, total_load)
        elif self.draft_method == "clamp":
            draft_limit = self._generate_draft_limit_by_clamp(batch_size, total_load)

        if self.normalize:
            demand = demand / total_load
            draft_limit = draft_limit / total_load

        td = TensorDict(
            {"locs": locs, "demand": demand, "draft_limit": draft_limit},
            batch_size=batch_size,
        )

        return td

    def _generate_draft_limit_by_clamp(self, batch_size, total_load):
        # 1. Determine constrained ports count
        constrain_pct = {"hard": 0.9, "medium": 0.75, "easy": 0.5}[self.hardness]
        num_constrained = int(constrain_pct * (self.num_loc + 1))

        # 2. Initialize all limits to maximum (total_load)
        draft_limit = torch.full((*batch_size, self.num_loc + 1), total_load, dtype=torch.float)

        # 3. Single-line constrained port selection (batch-friendly)
        selected_ports = torch.rand((*batch_size, self.num_loc)).topk(num_constrained, dim=-1).indices + 1

        # 4. Generate and sort constrained limits
        constrained_values = torch.randint(1, total_load + 1, (*batch_size, num_constrained))
        sorted_vals, _ = torch.sort(constrained_values, descending=True)

        # 5. Enforce feasibility: l_i >= n - i + 1 for sorted positions
        positions = torch.arange(1, num_constrained + 1, device=sorted_vals.device)
        min_required = (self.num_loc - positions + 1).expand_as(sorted_vals)
        clamped_vals = torch.maximum(sorted_vals, min_required)

        # 6. Scatter clamped values to selected ports
        draft_limit.scatter_(-1, selected_ports, clamped_vals.float())

        return draft_limit

    def _generate_draft_limit_by_rejection(self, batch_size, total_load):
        """Legacy method integration with naming alignment"""
        # Parameter mapping
        constrain_pct = {"hard": 0.9, "medium": 0.75, "easy": 0.5}[self.hardness]
        num_constrained = int(constrain_pct * (self.num_loc + 1))

        # Initialize all limits to max load
        draft_limit = torch.full((*batch_size, self.num_loc + 1), total_load, dtype=torch.float)

        # Batch processing with numpy compatibility
        for i in range(batch_size[0]):
            # Port selection (excluding depot)
            selected_ports = np.random.choice(range(1, self.num_loc + 1), num_constrained, replace=False)
            # Feasibility-ensured generation
            feasible = False
            while not feasible:
                constrained_limits = torch.randint(1, total_load, (num_constrained,))
                cnt = torch.bincount(constrained_limits, minlength=total_load + 1)
                cum_counts = torch.cumsum(cnt, dim=0)
                feasible = (cum_counts <= torch.arange(cum_counts.size(0))).all()

            # Assign values to current batch instance
            draft_limit[i][selected_ports] = constrained_limits.float()

        return draft_limit


def is_tspdl_feasible(td: TensorDict) -> torch.Tensor:
    """
    Validate TSPDL feasibility based on original problem definition:
    1. Ports (excluding depot) are sorted descendingly by draft limits
    2. Check if cumulative load L_i = (total_demand - sum_{j=1}^{i-1} d_j) ≤ l_i for all ports

    Args:
        td: TensorDict containing:
            - "demand": [batch_size, num_nodes] (depot demand at index 0)
            - "draft_limit": [batch_size, num_nodes] (depot limit at index 0)

    Returns:
        Boolean tensor [batch_size] indicating feasibility
    """
    # Extract key components
    demand, draft_limit = td["demand"], td["draft_limit"]
    # ===== 1. Sort ports (non-depot) by draft limits descendingly =====
    # Extract port components (indices 1~num_nodes-1)
    port_limits = draft_limit[:, 1:]  # [B, P]
    port_demand = demand[:, 1:]  # [B, P]

    # Sort ports by draft limits
    sorted_limits, sort_idx = torch.sort(port_limits, dim=-1, descending=True)  # [B, P]

    # Gather corresponding demand using sort indices
    sorted_demand = torch.gather(port_demand, -1, sort_idx)  # [B, P]

    # ===== 2. Compute cumulative loads L_i = total_demand - prefix_sum =====
    total_demand = port_demand.sum(dim=-1, keepdim=True)  # [B, 1]
    prefix_sum = torch.cumsum(sorted_demand, dim=-1)  # [B, P]
    load_on_arrival = total_demand - torch.cat(
        [
            torch.zeros_like(prefix_sum[:, :1]),  # L_1 = total_demand - 0
            prefix_sum[:, :-1],  # L_i = total_demand - sum_{1}^{i-1}
        ],
        dim=-1,
    )  # [B, P]

    # ===== 3. Validate load ≤ draft limit for all ports =====
    return (load_on_arrival <= sorted_limits).all(dim=-1)
