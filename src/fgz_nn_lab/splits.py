from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TemporalFold:
    fold_name: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class FoldScaler:
    """
    Leakage-safe fold scaler with:
      1. Sparse-archive channel guard (< min_valid_samples non-zero training points -> masked 0)
      2. 3-way Observer Reset Clock Selection per station strictly on train_idx:
         - Min T candidates: [0: nocturnal_cal, 1: morn24_coop, 2: eve24_coop]
         - Max T candidates: [3: afternoon_cal, 4: morn24_coop, 5: eve24_coop]
    """

    def __init__(self, min_valid_samples: int = 120, clip_zscore: float = 8.0) -> None:
        self.min_valid_samples = min_valid_samples
        self.clip_zscore = clip_zscore
        self.dyn_mean: np.ndarray | None = None
        self.dyn_std: np.ndarray | None = None
        self.active_channel_mask: np.ndarray | None = None
        self.ctx_mean: np.ndarray | None = None
        self.ctx_std: np.ndarray | None = None
        self.station_min_mode: dict[str, int] = {}  # 0, 1, or 2
        self.station_max_mode: dict[str, int] = {}  # 3, 4, or 5
        self.station_anchor_bias: dict[str, np.ndarray] = {}
        self.global_anchor_bias: np.ndarray = np.zeros((5, 2), dtype=np.float32)

    def fit(
        self,
        x_dyn: np.ndarray,
        mask_dyn: np.ndarray,
        x_ctx: np.ndarray,
        raw_anchor: np.ndarray,
        y_true: np.ndarray,
        station_ids: np.ndarray,
        raw_anchor_candidates: np.ndarray | None = None,
    ) -> "FoldScaler":
        valid_counts = mask_dyn.sum(axis=(0, 1))
        nonzero_counts = ((np.abs(x_dyn) > 1e-6) & (mask_dyn > 0.5)).sum(axis=(0, 1))

        safe_denom = np.maximum(valid_counts, 1.0)
        mean_raw = ((x_dyn * mask_dyn).sum(axis=(0, 1)) / safe_denom).astype(np.float32)
        sq_diff = (((x_dyn - mean_raw[None, None, :]) * mask_dyn) ** 2).sum(axis=(0, 1))
        std_raw = np.sqrt(sq_diff / safe_denom).astype(np.float32)

        self.active_channel_mask = (
            (valid_counts >= self.min_valid_samples)
            & (nonzero_counts >= min(40, self.min_valid_samples // 2))
            & (std_raw >= 1e-3)
        ).astype(np.float32)

        self.dyn_mean = mean_raw
        self.dyn_std = np.maximum(std_raw, 0.05).astype(np.float32)

        self.ctx_mean = x_ctx.mean(axis=0).astype(np.float32)
        self.ctx_std = np.maximum(x_ctx.std(axis=0), 0.05).astype(np.float32)

        selected_tr = raw_anchor.copy().astype(np.float32)
        if raw_anchor_candidates is not None and raw_anchor_candidates.shape[-1] == 6:
            for sid in np.unique(station_ids):
                m = station_ids == sid
                s_key = str(sid)

                # Best Min window (0=cal, 1=morn24, 2=eve24)
                y_min_s = y_true[m, :, 0]
                best_min_idx = 0
                best_min_mae = float(np.abs((y_min_s - raw_anchor_candidates[m, :, 0]) - np.median(y_min_s - raw_anchor_candidates[m, :, 0])).mean())
                for cand_i in (1, 2):
                    err_c = y_min_s - raw_anchor_candidates[m, :, cand_i]
                    mae_c = float(np.abs(err_c - np.median(err_c)).mean())
                    if mae_c + 0.08 < best_min_mae:
                        best_min_mae = mae_c
                        best_min_idx = cand_i
                self.station_min_mode[s_key] = best_min_idx
                selected_tr[m, :, 0] = raw_anchor_candidates[m, :, best_min_idx]

                # Best Max window (3=cal, 4=morn24, 5=eve24)
                y_max_s = y_true[m, :, 1]
                best_max_idx = 3
                best_max_mae = float(np.abs((y_max_s - raw_anchor_candidates[m, :, 3]) - np.median(y_max_s - raw_anchor_candidates[m, :, 3])).mean())
                for cand_i in (4, 5):
                    err_c = y_max_s - raw_anchor_candidates[m, :, cand_i]
                    mae_c = float(np.abs(err_c - np.median(err_c)).mean())
                    if mae_c + 0.08 < best_max_mae:
                        best_max_mae = mae_c
                        best_max_idx = cand_i
                self.station_max_mode[s_key] = best_max_idx
                selected_tr[m, :, 1] = raw_anchor_candidates[m, :, best_max_idx]

        err = y_true - selected_tr
        self.global_anchor_bias = np.median(err, axis=0).astype(np.float32)
        for sid in np.unique(station_ids):
            m = station_ids == sid
            if m.sum() >= 20:
                self.station_anchor_bias[str(sid)] = np.median(err[m], axis=0).astype(np.float32)
        return self

    def transform_inputs(
        self,
        x_dyn: np.ndarray,
        mask_dyn: np.ndarray,
        x_ctx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.dyn_mean is not None and self.dyn_std is not None
        assert self.active_channel_mask is not None
        assert self.ctx_mean is not None and self.ctx_std is not None

        eff_mask = (mask_dyn * self.active_channel_mask[None, None, :]).astype(np.float32)
        z_dyn = ((x_dyn - self.dyn_mean[None, None, :]) / self.dyn_std[None, None, :]) * eff_mask
        z_dyn = np.clip(z_dyn, -self.clip_zscore, self.clip_zscore).astype(np.float32)

        z_ctx = np.clip(
            (x_ctx - self.ctx_mean[None, :]) / self.ctx_std[None, :],
            -self.clip_zscore,
            self.clip_zscore,
        ).astype(np.float32)
        return z_dyn, eff_mask, z_ctx

    def calibrate_anchor(
        self,
        raw_anchor: np.ndarray,
        station_ids: np.ndarray,
        raw_anchor_candidates: np.ndarray | None = None,
    ) -> np.ndarray:
        out = raw_anchor.copy().astype(np.float32)
        for i, sid in enumerate(station_ids):
            s_key = str(sid)
            if raw_anchor_candidates is not None and raw_anchor_candidates.shape[-1] == 6:
                min_i = self.station_min_mode.get(s_key, 0)
                max_i = self.station_max_mode.get(s_key, 3)
                out[i, :, 0] = raw_anchor_candidates[i, :, min_i]
                out[i, :, 1] = raw_anchor_candidates[i, :, max_i]
            bias = self.station_anchor_bias.get(s_key, self.global_anchor_bias)
            out[i] = out[i] + bias
        return out


def make_purged_expanding_splits(
    issue_dates: np.ndarray,
    test_days: int = 330,
    val_folds: int = 2,
    val_fold_days: int = 240,
    purge_days: int = 6,
) -> list[TemporalFold]:
    unique_days = np.sort(np.unique(issue_dates))
    total_span_days = int((unique_days[-1] - unique_days[0]) / np.timedelta64(1, "D")) + 1

    if total_span_days < (test_days + val_folds * val_fold_days + 220):
        eff_test_days = min(test_days, max(90, int(total_span_days * 0.25)))
        eff_val_days = min(val_fold_days, max(80, int(total_span_days * 0.18)))
        eff_folds = 1 if total_span_days < 850 else val_folds
    else:
        eff_test_days = test_days
        eff_val_days = val_fold_days
        eff_folds = val_folds

    max_day = unique_days[-1]
    test_start = max_day - np.timedelta64(eff_test_days - 1, "D")
    test_idx = np.where(issue_dates >= test_start)[0]

    folds: list[TemporalFold] = []
    for k in range(eff_folds, 0, -1):
        val_end = test_start - np.timedelta64(purge_days + (k - 1) * eff_val_days + 1, "D")
        val_start = val_end - np.timedelta64(eff_val_days - 1, "D")
        train_end = val_start - np.timedelta64(purge_days + 1, "D")

        tr_idx = np.where(issue_dates <= train_end)[0]
        va_idx = np.where((issue_dates >= val_start) & (issue_dates <= val_end))[0]

        if len(tr_idx) >= 100 and len(va_idx) >= 50:
            folds.append(
                TemporalFold(
                    fold_name=f"fold_{eff_folds - k + 1}",
                    train_idx=tr_idx,
                    val_idx=va_idx,
                    test_idx=test_idx,
                )
            )

    if not folds:
        n_days = len(unique_days)
        t_cut = unique_days[int(n_days * 0.78)]
        v_cut = unique_days[int(n_days * 0.58)]
        folds.append(
            TemporalFold(
                fold_name="fold_compact",
                train_idx=np.where(issue_dates < (v_cut - np.timedelta64(purge_days, "D")))[0],
                val_idx=np.where((issue_dates >= v_cut) & (issue_dates < (t_cut - np.timedelta64(purge_days, "D"))))[0],
                test_idx=np.where(issue_dates >= t_cut)[0],
            )
        )
    return folds