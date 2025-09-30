import torch
import torch.nn as nn
from rl4co.utils.ops import gather_by_index


def get_one_hot_feature(index, num_classes):
    """
    Convert a tensor to one-hot encoding.
    Args:
        src (torch.Tensor): The input tensor.
        num_classes (int): The number of classes for one-hot encoding.
    Returns:
        torch.Tensor: One-hot encoded tensor.
    """
    one_hot = torch.zeros((*index.shape, num_classes), device=index.device, dtype=torch.float32)
    clamped_index = torch.clamp(index, 0, num_classes - 1).long()

    return one_hot.scatter(-1, clamped_index.unsqueeze(-1), 1.0)


class TSPDLInitEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 4):
        # node_dim = 4 : x, y, demand, draft_limit
        super(TSPDLInitEmbedding, self).__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim, bias=linear_bias)
        self.depot_embed = nn.Linear(2, embed_dim, bias=linear_bias)

    def forward(self, td):
        depot, ports = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        demand_sum = td["demand"].sum(dim=-1, keepdim=True)  # [B, 1]
        demand, draft_limit = (td["demand"][:, 1:] / demand_sum).unsqueeze(-1), (td["draft_limit"][:, 1:] / demand_sum).unsqueeze(-1)
        # [B, 1, 2]->[B, 1, embed_dim], [B, n, 4]->[B, n, embed_dim]
        depot_embedding = self.depot_embed(depot)
        node_embeddings = self.init_embed(torch.cat([ports, demand, draft_limit], dim=-1))
        return torch.cat([depot_embedding, node_embeddings], dim=-2)


class TSPDLContext(nn.Module):
    def __init__(self, embed_dim, linear_bias=True):
        super(TSPDLContext, self).__init__()
        self.embed_dim = embed_dim
        step_context_dim = embed_dim + 1
        self.project_context = nn.Linear(step_context_dim, embed_dim, bias=linear_bias)

    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])  # [B, S, h]
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        demand_sum = td["demand"].sum(dim=-1)  # [B,] or [B, S]
        state_embedding = (td["current_load"] / demand_sum).unsqueeze(-1)  # [B, 1] or [B, S, 1]
        return state_embedding

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        step_context_embedding = torch.cat([cur_node_embedding, state_embedding], dim=-1)
        return self.project_context(step_context_embedding)


class TSPDLRIEContext(nn.Module):
    def __init__(self, embed_dim, linear_bias=True, num_intensity_classes=5):
        super(TSPDLRIEContext, self).__init__()
        self.embed_dim = embed_dim
        step_context_dim = embed_dim + num_intensity_classes + 3
        self.project_context = nn.Linear(step_context_dim, embed_dim, bias=linear_bias)
        self.num_intensity_classes = num_intensity_classes

    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"].clone())  # [B, S, h]
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        demand_sum = td["demand"].sum(dim=-1)  # [B,] or [B, S]
        state_embedding = (td["current_load"] / demand_sum).unsqueeze(-1)  # [B, 1] or [B, S, 1]
        return state_embedding

    def _refinement_intensity_embedding(self, embeddings, td):
        revisit_count = gather_by_index(td["refine_intensity_stack"], td["step_idx"], dim=-1)  # [B,] or [B, S]
        one_hot_revisit_count = get_one_hot_feature(revisit_count, self.num_intensity_classes)
        one_hot_backtrack_budget_reached = get_one_hot_feature(td["backtrack_budget_reached"].float(), 2)

        return torch.cat([one_hot_revisit_count, one_hot_backtrack_budget_reached], dim=-1)

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        refinement_intensity_embed = self._refinement_intensity_embedding(embeddings, td)
        step_context_embedding = torch.cat([cur_node_embedding, state_embedding, refinement_intensity_embed], dim=-1)
        return self.project_context(step_context_embedding)


class TSPTWInitEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 4, tw_normalize=True, loc_scalar=1.0):
        # node_dim = 4 : x, y, tw_early, tw_late, temporarily service time is set to zero for all customers
        super(TSPTWInitEmbedding, self).__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim, bias=linear_bias)
        self.depot_embed = nn.Linear(2, embed_dim, bias=linear_bias)
        self.tw_normalize = tw_normalize
        self.loc_scalar = loc_scalar
        
    def forward(self, td):
        depot, customers = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        time_windows = td["time_windows"][:, 1:] / td["time_windows"][:, :1, 1:] if self.tw_normalize else td["time_windows"][:, 1:]
        
        depot, customers, time_windows = depot / self.loc_scalar, customers / self.loc_scalar, time_windows / self.loc_scalar
        
        # [B, 1, 2]->[B, 1, embed_dim], [B, n, 4]->[B, n, embed_dim]
        depot_embedding = self.depot_embed(depot)
        node_embeddings = self.init_embed(torch.cat([customers, time_windows], dim=-1))
        return torch.cat([depot_embedding, node_embeddings], dim=-2)


class TSPTWContext(nn.Module):
    def __init__(self, embed_dim, linear_bias=True, tw_normalize=True):
        super(TSPTWContext, self).__init__()
        self.embed_dim = embed_dim
        step_context_dim = embed_dim + 1
        self.project_context = nn.Linear(step_context_dim, embed_dim, bias=linear_bias)
        self.tw_normalize = tw_normalize

    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"].clone())  # [B, S, h]
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        curr_time = td["current_time"].unsqueeze(-1)  # [B, 1] or [B, S, 1]
        state_embedding = curr_time / td["time_windows"][..., 0, 1:] if self.tw_normalize else curr_time
        return state_embedding

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        step_context_embedding = torch.cat([cur_node_embedding, state_embedding], dim=-1)
        return self.project_context(step_context_embedding)


class TSPTWRIEContext(nn.Module):
    def __init__(self, embed_dim, linear_bias=True, tw_normalize=True, num_intensity_classes=5):
        super(TSPTWRIEContext, self).__init__()
        self.embed_dim = embed_dim
        step_context_dim = embed_dim + num_intensity_classes + 3
        self.project_context = nn.Linear(step_context_dim, embed_dim, bias=linear_bias)
        self.tw_normalize = tw_normalize
        self.num_intensity_classes = num_intensity_classes

    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"].clone())  # [B, S, h]
        return cur_node_embedding

    def _refinement_intensity_embedding(self, embeddings, td):
        revisit_count = gather_by_index(td["refine_intensity_stack"], td["step_idx"], dim=-1)  # [B,] or [B, S]
        one_hot_revisit_count = get_one_hot_feature(revisit_count, self.num_intensity_classes)
        one_hot_backtrack_budget_reached = get_one_hot_feature(td["backtrack_budget_reached"].float(), 2)
        return torch.cat([one_hot_revisit_count, one_hot_backtrack_budget_reached], dim=-1)

    def _state_embedding(self, embeddings, td):
        curr_time = td["current_time"].unsqueeze(-1)  # [B, 1] or [B, S, 1]
        state_embedding = curr_time / td["time_windows"][..., 0, 1:] if self.tw_normalize else curr_time
        return state_embedding

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        refinement_intensity_embed = self._refinement_intensity_embedding(embeddings, td)
        step_context_embedding = torch.cat([cur_node_embedding, state_embedding, refinement_intensity_embed], dim=-1)
        return self.project_context(step_context_embedding)
