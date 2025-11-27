import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadRelationAwareAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.d_k = self.d_model // self.n_heads
        self.device = args.device
        # self.gamma = args.gamma

        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(args.dropout)



    def forward(self, x):

        H = self.n_heads
        D = self.d_k  #

        #
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, d_model]
        B, L, _ = x.shape

        #
        Q = self.q_proj(x).view(B, H, D)
        K = self.k_proj(x).view(B, H, D)
        V = self.v_proj(x).view(B, H, D)
        attn_scores = torch.einsum("ihd,jhd->hij", Q, K) / (D ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.einsum("hij,jhd->ihd", attn_weights, V)
        attn_output = attn_output.contiguous().view(B, -1)
        output = self.out_proj(attn_output)
        return output




class MultiScaleRelationAwareEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.num_layers = args.num_layers
        self.layers = nn.ModuleList([
            MultiHeadRelationAwareAttention(args) for _ in range(self.num_layers)
        ])
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, time_emb=None):
        x = x.to(self.device)

        multi_scale_feats = []
        for layer in self.layers:
            residual = x
            x = F.relu(layer(x))  # No relation matrix passed anymore
            x = self.dropout(x)
            x = x + residual
            multi_scale_feats.append(x)

        fused = torch.cat(multi_scale_feats, dim=-1)

        if time_emb is not None:
            fused = fused + time_emb  # Adding time step embedding

        return fused


class Diffusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.time_steps = args.time_steps
        self.scheduler = args.scheduler
        self.s = args.s
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end

        if self.scheduler == "cosine":
            self.betas = self._cosine_beta_schedule()
        elif self.scheduler == "linear":
            self.betas = self._linear_beta_schedule(self.beta_start, self.beta_end)
        else:
            raise ValueError(f"Invalid scheduler: {self.scheduler}")

        self.alpha = (1 - self.betas).to(self.device)
        self.gamma = torch.cumprod(self.alpha, dim=0).to(self.device)

    def _cosine_beta_schedule(self):
        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps).to(self.device)
        alphas_cumprod = (
            torch.cos(((x / self.time_steps) + self.s) / (1 + self.s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0, 0.999)

    def _linear_beta_schedule(self, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, self.time_steps).to(self.device)

    def sample_time_steps(self, shape):
        return torch.randint(0, self.time_steps, shape, device=self.device)

    def noise(self, x, t):
        noise = torch.randn_like(x).to(self.device)
        gamma_t = self.gamma[t].unsqueeze(-1)
        noisy_x = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * noise
        return noisy_x, noise

    def forward(self, x):
        t = self.sample_time_steps(x.shape[:1])
        noisy_x, noise = self.noise(x, t)
        return noisy_x, noise, t


class DiffusionMultiScaleModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.diffusion = Diffusion(args).to(self.device)
        self.encoder = MultiScaleRelationAwareEncoder(args).to(self.device)

        self.input_proj = nn.Linear(args.total, args.d_model).to(self.device)
        self.norm = nn.LayerNorm(args.d_model).to(self.device)

        self.time_embedding = nn.Embedding(args.time_steps, args.d_model * args.num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(args.d_model * args.num_layers, args.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.mlp_hidden, 1),
            # nn.Sigmoid()
        ).to(self.device)

        self.noise_predictor = nn.Sequential(
            nn.Linear(args.d_model * args.num_layers, args.d_model),
            nn.ReLU(),
            nn.Linear(args.d_model, args.d_model)
        ).to(self.device)

    def forward(self, id, m_embed, d_embed):
        m_embed=m_embed.to(self.device)
        d_embed=d_embed.to(self.device)
        """
        id: List or tensor of (miRNA_idx, drug_idx) pairs, length = batch size
        m_embed: [n_miRNA, embed_dim]
        d_embed: [n_drug, embed_dim]
        rel_matrix: [batch, ...]
        """
        #
        if isinstance(id, list):
            miRNA_idx = torch.tensor([p[0] for p in id], dtype=torch.long, device=self.device)
            drug_idx = torch.tensor([p[1] for p in id], dtype=torch.long, device=self.device)
        else:

            miRNA_idx = id[:, 0].to(self.device)
            drug_idx = id[:, 1].to(self.device)

        #
        miRNA_feat = m_embed[miRNA_idx].to(self.device)  # [batch, embed_dim]
        drug_feat = d_embed[drug_idx].to(self.device)  # [batch, embed_dim]


        x = torch.cat([miRNA_feat, drug_feat], dim=1)  # [batch, total_dim]



        assert x.shape[-1] == self.input_proj.in_features, (
            f"Expected input dim {self.input_proj.in_features}, got {x.shape[-1]}"
        )


        x = x.float()


        x = self.input_proj(x)
        x = self.norm(x)

        # diffusion
        noisy_x, noise, t = self.diffusion(x)


        time_emb = self.time_embedding(t).to(self.device)

        # encoder
        encoded = self.encoder(noisy_x, time_emb=time_emb)

        preds = self.mlp(encoded).squeeze(-1)
        noise_pred = self.noise_predictor(encoded)

        return preds, noise_pred, noise, t

