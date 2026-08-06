"""
Stage 0 re-diagnosis: model architectures.

`AuthorityTemporalConflictGraph` and `ATCGPretrainer` are a faithful port of
the reference notebook's cells 6 and 17 (same layer shapes, same forward
path, same 1-dimensional bottleneck: node_projection -> ATCG ->
conflict_score in (0,1) via Sigmoid -> coherence_head = Linear(1,1) ->
BCEWithLogits).

`DirectHeadPretrainer` is the H1 ablation: same backbone + node_projection
feature extractor, but the downstream head consumes the full concatenated
pair representation [h_N ; h_D ; |h_N - h_D| ; h_N * h_D] (1024-dim) through
an MLP directly to a single logit -- no ATCG graph-attention module, no
scalar squeeze before the loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AuthorityTemporalConflictGraph(nn.Module):
    """Verbatim port of cell 6 of the reference notebook."""

    def __init__(self, node_dim=256, hidden_dim=256, num_heads=4):
        super().__init__()
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.authority_matrix = nn.Parameter(torch.tensor([
            [1.0, 0.5],
            [1.5, 1.0],
        ]))

        self.query = nn.Linear(node_dim, hidden_dim)
        self.key = nn.Linear(node_dim, hidden_dim)
        self.value = nn.Linear(node_dim, hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, node_dim)
        self.layer_norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(0.1)

        self.conflict_scorer = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(node_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, nursing_node, discharge_node, nursing_role=0, discharge_role=1):
        batch_size = nursing_node.shape[0]
        nodes = torch.stack([nursing_node, discharge_node], dim=1)
        auth_weight = self.authority_matrix[nursing_role, discharge_role]

        Q = self.query(nodes)
        K = self.key(nodes)
        V = self.value(nodes)

        Q = Q.view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores * auth_weight
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, 2, -1)
        attended = self.output_proj(attended)
        attended = self.layer_norm(nodes + attended)

        nursing_attended = attended[:, 0, :]
        discharge_attended = attended[:, 1, :]

        edge_repr = torch.cat([nursing_attended, discharge_attended], dim=-1)
        conflict_score = self.conflict_scorer(edge_repr)

        return edge_repr, conflict_score, attn_weights


class ATCGPretrainer(nn.Module):
    """Verbatim port of cell 17's ATCGPretrainer -- the H1 bottleneck
    architecture used in the original (failed) Stage 0 run."""

    def __init__(self, backbone, hidden_size=768, cde_dim=256):
        super().__init__()
        self.backbone = backbone
        self.atcg = AuthorityTemporalConflictGraph(node_dim=cde_dim)

        self.node_projection = nn.Sequential(
            nn.Linear(hidden_size, cde_dim),
            nn.LayerNorm(cde_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.coherence_head = nn.Linear(1, 1)

    def encode(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.node_projection(cls)

    def forward(self, nursing_input_ids, nursing_attention_mask,
                discharge_input_ids, discharge_attention_mask, labels=None):
        nursing_node = self.encode(nursing_input_ids, nursing_attention_mask)
        discharge_node = self.encode(discharge_input_ids, discharge_attention_mask)

        edge_repr, conflict_score, attn_weights = self.atcg(nursing_node, discharge_node)
        coherence_logit = self.coherence_head(conflict_score).squeeze(-1)

        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(coherence_logit, labels)
        return loss, coherence_logit, conflict_score


class DirectHeadPretrainer(nn.Module):
    """H1 ablation. Same backbone + node_projection feature extractor as
    ATCGPretrainer, but the downstream head sees the full 1024-dim
    concatenated pair representation directly -- no ATCG module, no
    single-scalar squeeze before the loss."""

    def __init__(self, backbone, hidden_size=768, cde_dim=256):
        super().__init__()
        self.backbone = backbone

        self.node_projection = nn.Sequential(
            nn.Linear(hidden_size, cde_dim),
            nn.LayerNorm(cde_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(cde_dim * 4, cde_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cde_dim, 1),
        )

    def encode(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.node_projection(cls)

    def forward(self, nursing_input_ids, nursing_attention_mask,
                discharge_input_ids, discharge_attention_mask, labels=None):
        h_n = self.encode(nursing_input_ids, nursing_attention_mask)
        h_d = self.encode(discharge_input_ids, discharge_attention_mask)

        pair_repr = torch.cat([h_n, h_d, torch.abs(h_n - h_d), h_n * h_d], dim=-1)
        logit = self.classifier(pair_repr).squeeze(-1)

        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logit, labels)
        return loss, logit, None
