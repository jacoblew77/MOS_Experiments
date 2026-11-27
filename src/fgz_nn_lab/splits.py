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
    Leakage-safe fold scaler with two critical safeguards:
      1. Sparse-Archive Channel Guard: Any dynamic channel with < min_valid_samples non-zero
         observations in train_idx is permanently masked out (mask=0, x=0) across train/val/test
         so late-starting archives (e.g., HRRR starting Sep 2025) never explode test z-scores.
      2. COOP Max-Window Phase Selection: Compares bias-adjusted MAE of calendar afternoon vs
         24-hour morning-reset window per station strictly on train_idx.
    """

    def __init__(self, min_valid_samples: int = 150, clip_zscore: float = 8.0) -> None:
        self.min_valid_samples = min_valid_samples
        self.clip_zscore = clip_zscore
        self.dyn_mean: np.ndarray | None = None
        self.dyn_std: np.ndarray | None = None
        self.active_channel_mask: np.ndarray | None = None
        self.ctx_mean: np.ndarray | None = None
        self.ctx_std: np.ndarray | None = None
        self.station_max_mode: dict[str, int] = {}  # 1 = calendar_afternoon, 2 = coop24_morning_reset
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
        dyn_sum = (x_dyn * mask_dyn).sum(axis=(0, 1))
        mean_raw = (dyn_sum / safe_denom).astype(np.float32)

        sq_diff = (((x_dyn - mean_raw[None, None, :]) * mask_dyn) ** 2).sum(axis=(0, 1))
        std_raw = np.sqrt(sq_diff / safe_denom).astype(np.float32)

        # Require both sufficient valid count AND genuine non-zero variance in the training fold
        self.active_channel_mask = (
            (valid_counts >= self.min_valid_samples)
            & (nonzero_counts >= min(50, self.min_valid_samples // 2))
            & (std_raw >= 1e-3)
        ).astype(np.float32)

        self.dyn_mean = mean_raw
        self.dyn_std = np.maximum(std_raw, 0.05).astype(np.float32)

        self.ctx_mean = x_ctx.mean(axis=0).astype(np.float32)
        self.ctx_std = np.maximum(x_ctx.std(axis=0), 0.05).astype(np.float32)

        # Select optimal Max-T COOP window mode per station strictly on training data
        selected_tr = raw_anchor.copy().astype(np.float32)
        if raw_anchor_candidates is not None:
            for sid in np.unique(station_ids):
                m = station_ids == sid
                y_max_s = y_true[m, :, 1]
                cand_cal = raw_anchor_candidates[m, :, 1]
                cand_c24 = raw_anchor_candidates[m, :, 2]

                err_cal = y_max_s - cand_cal
                err_c24 = y_max_s - cand_c24
                mae_cal = float(np.abs(err_cal - np.median(err_cal)).mean())
                mae_c24 = float(np.abs(err_c24 - np.median(err_c24)).mean())

                # Switch to 24h morning-reset window when it clearly improves training MAE
                if mae_c24 + 0.10 < mae_cal:
                    self.station_max_mode[str(sid)] = 2
                    selected_tr[m, :, 1] = cand_c24
                else:
                    self.station_max_mode[str(sid)] = 1
                    selected_tr[m, :, 1] = cand_cal
                selected_tr[m, :, 0] = raw_anchor_candidates[m, :, 0]

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
            if raw_anchor_candidates is not None:
                mode_idx = self.station_max_mode.get(s_key, 1)
                out[i, :, 0] = raw_anchor_candidates[i, :, 0]
                out[i, :, 1] = raw_anchor_candidates[i, :, mode_idx]
            bias = self.station_anchor_bias.get(s_key, self.global_anchor_bias)
            out[i] = out[i] + bias
        return out


def make_purged_expanding_splits(
    issue_dates: np.ndarray,
    test_days: int = 365,
    val_folds: int = 2,
    val_fold_days: int = 300,
    purge_days: int = 6,
) -> list[TemporalFold]:
    unique_days = np.sort(np.unique(issue_dates))
    total_span_days = int((unique_days[-1] - unique_days[0]) / np.timedelta64(1, "D")) + 1

    # Adapt fold lengths smoothly if archive span is ~2.8 years (e.g., true_lead_nwp 2024-2026)
    if total_span_days < (test_days + val_folds * val_fold_days + 240):
        eff_test_days = min(test_days, max(120, int(total_span_days * 0.25)))
        eff_val_days = min(val_fold_days, max(100, int(total_span_days * 0.18)))
        eff_folds = 1 if total_span_days < 900 else val_folds
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
        tr_end = v_cut - np.timedelta64(purge_days, "D")
        va_end = t_cut - np.timedelta64(purge_days, "D")
        folds.append(
            TemporalFold(
                fold_name="fold_compact",
                train_idx=np.where(issue_dates < tr_end)[0],
                val_idx=np.where((issue_dates >= v_cut) & (issue_dates < va_end))[0],
                test_idx=np.where(issue_dates >= t_cut)[0],
            )
        )
    return folds