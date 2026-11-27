from __future__ import annotations

import copy
import time
from datetime import datetime, timezone
from typing import Any

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


def fit_l1_optimal_blend_weights(
    pred_a_val: np.ndarray,
    pred_b_val: np.ndarray,
    y_val: np.ndarray,
) -> np.ndarray:
    weights = np.full((5, 2), 0.5, dtype=np.float32)
    if len(y_val) == 0:
        return weights
    grid = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    for d in range(5):
        for t_i in range(2):
            pa = pred_a_val[:, d, t_i][:, None]
            pb = pred_b_val[:, d, t_i][:, None]
            yt = y_val[:, d, t_i][:, None]
            blended = grid[None, :] * pa + (1.0 - grid[None, :]) * pb
            maes = np.abs(blended - yt).mean(axis=0)
            weights[d, t_i] = float(grid[int(np.argmin(maes))])
    return weights


def apply_blend_weights(pred_a: np.ndarray, pred_b: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = weights[None, :, :]
    return (w * pred_a + (1.0 - w) * pred_b).astype(np.float32)


def fit_network_aware_blend_weights(
    pred_nn_val: np.ndarray,
    pred_lgb_val: np.ndarray,
    y_val: np.ndarray,
    networks_val: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fits dedicated (5, 2) L1 stack weights separately for AZ_ASOS vs AZ_COOP on validation folds."""
    global_w = fit_l1_optimal_blend_weights(pred_nn_val, pred_lgb_val, y_val)
    out: dict[str, np.ndarray] = {"GLOBAL": global_w}
    for net in ("AZ_ASOS", "AZ_COOP"):
        m = networks_val == net
        if m.sum() >= 40:
            out[net] = fit_l1_optimal_blend_weights(pred_nn_val[m], pred_lgb_val[m], y_val[m])
        else:
            out[net] = global_w.copy()
    return out


def apply_network_aware_blend(
    pred_nn: np.ndarray,
    pred_lgb: np.ndarray,
    networks: np.ndarray,
    weight_map: dict[str, np.ndarray],
) -> np.ndarray:
    out = np.zeros_like(pred_nn, dtype=np.float32)
    g_w = weight_map.get("GLOBAL", np.full((5, 2), 0.5, dtype=np.float32))
    for i, net in enumerate(networks):
        w = weight_map.get(str(net), g_w)
        out[i] = w * pred_nn[i] + (1.0 - w) * pred_lgb[i]
    return out


def fit_conformal_residual_quantiles(
    pred_val: np.ndarray,
    y_val: np.ndarray,
    networks_val: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Calibrates empirical P10 & P90 out-of-fold error quantiles (y - pred) per network and lead day."""
    res_all = y_val - pred_val
    q_global = {
        "q10": np.percentile(res_all, 10, axis=0).astype(np.float32),
        "q90": np.percentile(res_all, 90, axis=0).astype(np.float32),
    }
    out: dict[str, dict[str, np.ndarray]] = {"GLOBAL": q_global}
    for net in ("AZ_ASOS", "AZ_COOP"):
        m = networks_val == net
        if m.sum() >= 40:
            r_n = res_all[m]
            out[net] = {
                "q10": np.percentile(r_n, 10, axis=0).astype(np.float32),
                "q90": np.percentile(r_n, 90, axis=0).astype(np.float32),
            }
        else:
            out[net] = q_global
    return out


def fit_predict_ridge_baseline(
    x_dyn_tr: np.ndarray,
    mask_tr: np.ndarray,
    x_ctx_tr: np.ndarray,
    anchor_tr: np.ndarray,
    y_tr: np.ndarray,
    x_dyn_ev: np.ndarray,
    mask_ev: np.ndarray,
    x_ctx_ev: np.ndarray,
    anchor_ev: np.ndarray,
) -> np.ndarray:
    preds = np.zeros_like(anchor_ev, dtype=np.float32)
    for d in range(5):
        f_tr = np.concatenate([x_dyn_tr[:, d, :], mask_tr[:, d, :], x_ctx_tr], axis=1)
        f_ev = np.concatenate([x_dyn_ev[:, d, :], mask_ev[:, d, :], x_ctx_ev], axis=1)
        for target_i in range(2):
            res_tr = y_tr[:, d, target_i] - anchor_tr[:, d, target_i]
            reg = Ridge(alpha=50.0)
            reg.fit(f_tr, res_tr)
            preds[:, d, target_i] = anchor_ev[:, d, target_i] + reg.predict(f_ev).astype(np.float32)
    return preds


def fit_predict_lgbm_baseline(
    x_dyn_tr: np.ndarray,
    mask_tr: np.ndarray,
    x_ctx_tr: np.ndarray,
    st_idx_tr: np.ndarray,
    anchor_tr: np.ndarray,
    y_tr: np.ndarray,
    x_dyn_va: np.ndarray,
    mask_va: np.ndarray,
    x_ctx_va: np.ndarray,
    st_idx_va: np.ndarray,
    anchor_va: np.ndarray,
    y_va: np.ndarray,
    x_dyn_ev: np.ndarray,
    mask_ev: np.ndarray,
    x_ctx_ev: np.ndarray,
    st_idx_ev: np.ndarray,
    anchor_ev: np.ndarray,
    seed: int = 42,
    fold_label: str = "",
) -> tuple[np.ndarray, np.ndarray, list[list[lgb.LGBMRegressor]]]:
    val_preds = np.zeros_like(anchor_va, dtype=np.float32)
    ev_preds = np.zeros_like(anchor_ev, dtype=np.float32)
    fitted_models: list[list[lgb.LGBMRegressor]] = []
    t0 = time.monotonic()

    for d in range(5):
        f_tr = np.concatenate([x_dyn_tr[:, d, :], mask_tr[:, d, :], x_ctx_tr, st_idx_tr[:, None], anchor_tr[:, d, :]], axis=1)
        f_va = np.concatenate([x_dyn_va[:, d, :], mask_va[:, d, :], x_ctx_va, st_idx_va[:, None], anchor_va[:, d, :]], axis=1)
        f_ev = np.concatenate([x_dyn_ev[:, d, :], mask_ev[:, d, :], x_ctx_ev, st_idx_ev[:, None], anchor_ev[:, d, :]], axis=1)
        day_models: list[lgb.LGBMRegressor] = []
        for target_i in range(2):
            res_tr = y_tr[:, d, target_i] - anchor_tr[:, d, target_i]
            res_va = y_va[:, d, target_i] - anchor_va[:, d, target_i]
            gbm = lgb.LGBMRegressor(
                objective="regression_l1",
                n_estimators=550,
                learning_rate=0.035,
                num_leaves=31,
                subsample_for_bin=50000,
                colsample_bytree=0.8,
                subsample=0.8,
                min_child_samples=25,
                random_state=seed + d * 10 + target_i,
                n_jobs=-1,
                verbose=-1,
            )
            gbm.fit(f_tr, res_tr, eval_set=[(f_va, res_va)], callbacks=[lgb.early_stopping(35, verbose=False)])
            val_preds[:, d, target_i] = anchor_va[:, d, target_i] + gbm.predict(f_va).astype(np.float32)
            ev_preds[:, d, target_i] = anchor_ev[:, d, target_i] + gbm.predict(f_ev).astype(np.float32)
            day_models.append(gbm)
        fitted_models.append(day_models)
        print(f"      [{_ts()}] [LGBM {fold_label}] Day {d + 1}/5 Min+Max trees fitted ({time.monotonic() - t0:.1f}s)", flush=True)
    return val_preds, ev_preds, fitted_models


def predict_lgbm_grid(
    models_grid: list[list[lgb.LGBMRegressor]],
    x_dyn: np.ndarray,
    mask_dyn: np.ndarray,
    x_ctx: np.ndarray,
    st_idx: np.ndarray,
    anchor: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(anchor, dtype=np.float32)
    for d in range(5):
        f_ev = np.concatenate([x_dyn[:, d, :], mask_dyn[:, d, :], x_ctx, st_idx[:, None], anchor[:, d, :]], axis=1)
        for t_i in range(2):
            out[:, d, t_i] = anchor[:, d, t_i] + models_grid[d][t_i].predict(f_ev).astype(np.float32)
    return out


class ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(self.conv1(h.transpose(1, 2)).transpose(1, 2))
        h = self.drop(h)
        h = self.norm2(h)
        h = self.conv2(h.transpose(1, 2)).transpose(1, 2)
        return x + self.drop(h)


class FGZMultiHorizonResNet(nn.Module):
    def __init__(
        self,
        dyn_dim: int,
        ctx_dim: int,
        num_stations: int = 32,
        hidden_dim: int = 96,
        ctx_embed_dim: int = 32,
        station_embed_dim: int = 12,
        dropout: float = 0.18,
        channel_dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.channel_dropout = channel_dropout
        in_per_day = dyn_dim * 2 + 2

        self.station_embed = nn.Embedding(max(num_stations, 2), station_embed_dim)
        self.ctx_encoder = nn.Sequential(
            nn.Linear(ctx_dim + station_embed_dim, ctx_embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(ctx_embed_dim, ctx_embed_dim),
            nn.SiLU(),
        )
        self.film_gamma = nn.Linear(ctx_embed_dim, hidden_dim)
        self.film_beta = nn.Linear(ctx_embed_dim, hidden_dim)

        self.day_embed = nn.Embedding(5, hidden_dim)
        self.in_proj = nn.Sequential(
            nn.Linear(in_per_day, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.tblock1 = ResidualTemporalBlock(hidden_dim, dropout)
        self.tblock2 = ResidualTemporalBlock(hidden_dim, dropout)
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.post_norm = nn.LayerNorm(hidden_dim + ctx_embed_dim)

        head_in = hidden_dim + ctx_embed_dim
        self.min_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.max_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x_dyn: torch.Tensor,
        mask_dyn: torch.Tensor,
        x_ctx: torch.Tensor,
        st_idx: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        if self.training and self.channel_dropout > 0.0:
            keep_prob = 1.0 - self.channel_dropout
            chan_mask = torch.bernoulli(
                torch.full((x_dyn.size(0), 1, x_dyn.size(2)), keep_prob, device=x_dyn.device)
            )
            x_dyn = x_dyn * chan_mask
            mask_dyn = mask_dyn * chan_mask

        anchor_scaled = (anchor - 50.0) / 25.0
        day_in = torch.cat([x_dyn, mask_dyn, anchor_scaled], dim=-1)

        h = self.in_proj(day_in)
        day_ids = torch.arange(5, device=x_dyn.device).unsqueeze(0)
        h = h + self.day_embed(day_ids)

        st_emb = self.station_embed(st_idx)
        ctx_emb = self.ctx_encoder(torch.cat([x_ctx, st_emb], dim=-1))
        gamma = 1.0 + 0.25 * torch.tanh(self.film_gamma(ctx_emb)).unsqueeze(1)
        beta = self.film_beta(ctx_emb).unsqueeze(1)
        h = gamma * h + beta

        h = self.tblock1(h)
        h = self.tblock2(h)
        gru_out, _ = self.bigru(h)
        h = h + gru_out

        ctx_rep = ctx_emb.unsqueeze(1).expand(-1, 5, -1)
        joint = self.post_norm(torch.cat([h, ctx_rep], dim=-1))

        delta_min = self.min_head(joint).squeeze(-1)
        delta_max = self.max_head(joint).squeeze(-1)
        return anchor + torch.stack([delta_min, delta_max], dim=-1)


def compute_extremum_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    horizon_weights: torch.Tensor,
    min_weight: float = 1.0,
    max_weight: float = 1.0,
    use_smooth_l1: bool = False,
    smooth_beta: float = 2.0,
    crossing_weight: float = 0.20,
) -> torch.Tensor:
    if use_smooth_l1:
        err_min = F.smooth_l1_loss(pred[:, :, 0], target[:, :, 0], beta=smooth_beta, reduction="none")
        err_max = F.smooth_l1_loss(pred[:, :, 1], target[:, :, 1], beta=smooth_beta, reduction="none")
    else:
        err_min = torch.abs(pred[:, :, 0] - target[:, :, 0])
        err_max = torch.abs(pred[:, :, 1] - target[:, :, 1])

    weighted = horizon_weights.unsqueeze(0) * (min_weight * err_min + max_weight * err_max)
    crossing = F.relu(pred[:, :, 0] - pred[:, :, 1] + 0.5).mean()
    return weighted.mean() + crossing_weight * crossing


def train_and_predict_nn(
    x_dyn_tr: np.ndarray,
    mask_tr: np.ndarray,
    x_ctx_tr: np.ndarray,
    st_idx_tr: np.ndarray,
    anchor_tr: np.ndarray,
    y_tr: np.ndarray,
    x_dyn_va: np.ndarray,
    mask_va: np.ndarray,
    x_ctx_va: np.ndarray,
    st_idx_va: np.ndarray,
    anchor_va: np.ndarray,
    y_va: np.ndarray,
    x_dyn_ev: np.ndarray,
    mask_ev: np.ndarray,
    x_ctx_ev: np.ndarray,
    st_idx_ev: np.ndarray,
    anchor_ev: np.ndarray,
    nn_cfg: dict[str, Any],
    num_stations: int = 32,
    seed: int = 42,
    fold_label: str = "",
    pretrained_state_dict: dict[str, Any] | None = None,
    override_epochs: int | None = None,
) -> tuple[np.ndarray, np.ndarray, FGZMultiHorizonResNet]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.monotonic()

    model = FGZMultiHorizonResNet(
        dyn_dim=x_dyn_tr.shape[2],
        ctx_dim=x_ctx_tr.shape[1],
        num_stations=num_stations,
        hidden_dim=int(nn_cfg.get("hidden_dim", 96)),
        ctx_embed_dim=int(nn_cfg.get("ctx_embed_dim", 32)),
        station_embed_dim=int(nn_cfg.get("station_embed_dim", 12)),
        dropout=float(nn_cfg.get("dropout", 0.18)),
        channel_dropout=float(nn_cfg.get("channel_dropout", 0.12)),
    ).to(device)

    is_finetune = pretrained_state_dict is not None
    if is_finetune:
        model.load_state_dict(pretrained_state_dict, strict=False)

    lr = float(nn_cfg.get("finetune_lr", 4e-4)) if is_finetune else float(nn_cfg.get("lr", 1e-3))
    max_epochs = override_epochs if override_epochs is not None else int(nn_cfg.get("max_epochs", 45))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(nn_cfg.get("weight_decay", 8e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    ds_tr = TensorDataset(
        torch.from_numpy(x_dyn_tr),
        torch.from_numpy(mask_tr),
        torch.from_numpy(x_ctx_tr),
        torch.from_numpy(st_idx_tr.astype(np.int64)),
        torch.from_numpy(anchor_tr),
        torch.from_numpy(y_tr),
    )
    loader_tr = DataLoader(ds_tr, batch_size=int(nn_cfg.get("batch_size", 128)), shuffle=True, drop_last=False)

    h_weights = torch.tensor(nn_cfg.get("horizon_weights", [1.0] * 5), dtype=torch.float32, device=device)
    warmup_epochs = 0 if is_finetune else int(nn_cfg.get("warmup_smooth_l1_epochs", 6))
    smooth_beta = float(nn_cfg.get("smooth_l1_beta", 2.0))
    patience = int(nn_cfg.get("patience", 8))

    t_va = (
        torch.from_numpy(x_dyn_va).to(device),
        torch.from_numpy(mask_va).to(device),
        torch.from_numpy(x_ctx_va).to(device),
        torch.from_numpy(st_idx_va.astype(np.int64)).to(device),
        torch.from_numpy(anchor_va).to(device),
    )
    y_va_t = torch.from_numpy(y_va).to(device)

    best_val_mae = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        use_smooth = epoch <= warmup_epochs
        for bx_dyn, bm_dyn, bx_ctx, bs_idx, b_anc, by in loader_tr:
            optimizer.zero_grad()
            pred = model(bx_dyn.to(device), bm_dyn.to(device), bx_ctx.to(device), bs_idx.to(device), b_anc.to(device))
            loss = compute_extremum_loss(
                pred, by.to(device), horizon_weights=h_weights,
                min_weight=float(nn_cfg.get("min_weight", 1.0)),
                max_weight=float(nn_cfg.get("max_weight", 1.0)),
                use_smooth_l1=use_smooth, smooth_beta=smooth_beta,
                crossing_weight=float(nn_cfg.get("crossing_penalty_weight", 0.20)),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_pred = model(*t_va)
            abs_diff = torch.abs(va_pred - y_va_t)
            val_mae = float(abs_diff.mean().item())
            val_min_mae = float(abs_diff[:, :, 0].mean().item())
            val_max_mae = float(abs_diff[:, :, 1].mean().item())

        improved = val_mae < best_val_mae - 1e-4
        if improved:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 5 == 0 or stale_epochs >= patience or epoch == max_epochs:
            tag = "*best*" if improved else f"stale {stale_epochs}/{patience}"
            print(
                f"      [{_ts()}] [NN {fold_label}] Ep {epoch:02d}/{max_epochs:02d} | "
                f"Val Min: {val_min_mae:.3f}F | Val Max: {val_max_mae:.3f}F | Combined: {val_mae:.3f}F ({tag}, {time.monotonic() - t0:.1f}s)",
                flush=True,
            )
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds = model(*t_va).cpu().numpy().astype(np.float32)
        ev_preds = model(
            torch.from_numpy(x_dyn_ev).to(device),
            torch.from_numpy(mask_ev).to(device),
            torch.from_numpy(x_ctx_ev).to(device),
            torch.from_numpy(st_idx_ev.astype(np.int64)).to(device),
            torch.from_numpy(anchor_ev).to(device),
        ).cpu().numpy().astype(np.float32)

    return val_preds, ev_preds, model