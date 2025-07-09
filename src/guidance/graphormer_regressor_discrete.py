import torch, torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete
from src.diffusion.diffusion_utils import sample_discrete_features
from src.GP5.models.graphormer import GraphormerModel   # 위에 정의한 nn.Module

class GraphormerRegressor(pl.LightningModule):
    def __init__(self, cfg, schema, num_steps=1000):
        super().__init__()
        self.save_hyperparameters()          # ckpt에 cfg 저장
        self.backbone = GraphormerModel(cfg["model"], target_type=cfg["target_type"])
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)
        self.schema = schema                 # encode_global용
        self.loss_fn = nn.L1Loss()           # 예: MAE

    # ---------- core helpers ----------
    def apply_noise(self, X0, E0, node_mask):
        t_int  = torch.randint(1, self.noise_sched.num_steps+1,
                               (X0.size(0),), device=self.device)
        Qt_bar = self.noise_sched.q_bar(t_int)            # transition matrices

        probX  = torch.einsum("bnd,dt->bnt", X0, Qt_bar["X"])
        probE  = torch.einsum("bnmd,dt->bnmt", E0, Qt_bar["E"])

        Xt, Et = sample_discrete_features(probX, probE, node_mask)
        return {"X_t": Xt, "E_t": Et, "t": t_int.float() / self.noise_sched.num_steps}

    def add_structural_feats(self, noisy, node_mask):
        X_t, E_t = noisy["X_t"], noisy["E_t"]
        idx_edge = E_t.argmax(-1)            # (B,N,N)
        adj      = idx_edge > 0

        in_deg   = adj.sum(-2)
        out_deg  = adj.sum(-1)
        spatial  = self._batch_floyd(adj)    # (B,N,N) long

        noisy.update({
            "in_degree": in_deg, "out_degree": out_deg,
            "spatial_pos": spatial, "attn_edge_type": idx_edge.unsqueeze(-1).long()
        })
        return noisy

    def _batch_floyd(self, adj, max_dist=5):
        B,N,_ = adj.shape
        inf = adj.new_full((B,N,N), 1e9).float()
        dist = torch.where(adj, 1., inf); dist.diagonal(dim1=1,dim2=2).fill_(0.)
        for k in range(N):
            dist = torch.minimum(dist, dist[:,:,k:k+1]+dist[:,k:k+1,:])
        return dist.clamp_max(max_dist).long()

    # ---------- Lightning interface ----------
    def training_step(self, batch, _):
        X0,E0,y0,node_mask = batch
        noisy = self.apply_noise(X0,E0,node_mask)
        noisy = self.add_structural_feats(noisy, node_mask)

        # extra: timestep t
        extra = {"t": noisy["t"].unsqueeze(-1)}

        # Graphormer 입력 dict
        gdata = {
            "x": noisy["X_t"].argmax(-1).long(),      # 정수 index (원하면 one-hot→float 그대로 넣어도 OK)
            "attn_edge_type": noisy["attn_edge_type"],
            "in_degree": noisy["in_degree"],
            "out_degree": noisy["out_degree"],
            "spatial_pos": noisy["spatial_pos"],
            "node_mask": node_mask,
        }
        pred = self.backbone(gdata, targets=y0)       # GraphormerModel.forward

        loss = self.loss_fn(pred, y0)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)
