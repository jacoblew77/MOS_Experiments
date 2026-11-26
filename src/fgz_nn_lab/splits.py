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
    """Standardizes valid (mask == 1) dynamic features and static context strictly on train_idx."""

    def __init__(self) -> None:
        self.dyn_mean: np.ndarray | None = None
        self.dyn_std: np.ndarray | None = None
        self.ctx_mean: np.ndarray | None = None
        self.ctx_std: np.ndarray | None = None
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
    ) -> "FoldScaler":
        # Masked mean & std across (N, 5) for each C_dyn channel
        valid_counts = np.maximum(mask_dyn.sum(axis=(0, 1)), 1.0)
        dyn_sum = (x_dyn * mask_dyn).sum(axis=(0, 1))
        self.dyn_mean = (dyn_sum / valid_counts).astype(np.float32)

        sq_diff = (((x_dyn - self.dyn_mean[None, None, :]) * mask_dyn) ** 2).sum(axis=(0, 1))
        self.dyn_std = np.maximum(np.sqrt(sq_diff / valid_counts), 1e-4).astype(np.float32)

        self.ctx_mean = x_ctx.mean(axis=0).astype(np.float32)
        self.ctx_std = np.maximum(x_ctx.std(axis=0), 1e-4).astype(np.float32)

        # Leak-free station-specific median bias correction on the anchor
        err = y_true - raw_anchor  # (N, 5, 2)
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
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.dyn_mean is not None and self.dyn_std is not None
        assert self.ctx_mean is not None and self.ctx_std is not None
        x_dyn_norm = ((x_dyn - self.dyn_mean[None, None, :]) / self.dyn_std[None, None, :]) * mask_dyn
        x_ctx_norm = (x_ctx - self.ctx_mean[None, :]) / self.ctx_std[None, :]
        return x_dyn_norm.astype(np.float32), x_ctx_norm.astype(np.float32)

    def calibrate_anchor(self, raw_anchor: np.ndarray, station_ids: np.ndarray) -> np.ndarray:
        out = raw_anchor.copy().astype(np.float32)
        for i, sid in enumerate(station_ids):
            bias = self.station_anchor_bias.get(str(sid), self.global_anchor_bias)
            out[i] = out[i] + bias
        return out


def make_purged_expanding_splits(
    issue_dates: np.ndarray,
    test_days: int = 365,
    val_folds: int = 2,
    val_fold_days: int = 365,
    purge_days: int = 6,
) -> list[TemporalFold]:
    """
    Creates chronological expanding-window validation folds + held-out test set.
    Enforces a strict `purge_days` embargo buffer between every split boundary so
    5-day forecast targets never cross train -> val or val -> test boundaries.
    """
    unique_days = np.sort(np.unique(issue_dates))
    max_day = unique_days[-1]

    test_start = max_day - np.timedelta64(test_days - 1, "D")
    test_idx = np.where(issue_dates >= test_start)[0]

    folds: list[TemporalFold] = []
    for k in range(val_folds, 0, -1):
        val_end = test_start - np.timedelta64(purge_days + (k - 1) * val_fold_days + 1, "D")
        val_start = val_end - np.timedelta64(val_fold_days - 1, "D")
        train_end = val_start - np.timedelta64(purge_days + 1, "D")

        tr_idx = np.where(issue_dates <= train_end)[0]
        va_idx = np.where((issue_dates >= val_start) & (issue_dates <= val_end))[0]

        if len(tr_idx) > 0 and len(va_idx) > 0:
            folds.append(
                TemporalFold(
                    fold_name=f"fold_{val_folds - k + 1}",
                    train_idx=tr_idx,
                    val_idx=va_idx,
                    test_idx=test_idx,
                )
            )

    if not folds:
        # Fallback single chronological split for shorter smoke-test windows
        n_days = len(unique_days)
        t_cut = unique_days[int(n_days * 0.80)]
        v_cut = unique_days[int(n_days * 0.60)]
        tr_end = v_cut - np.timedelta64(purge_days, "D")
        va_end = t_cut - np.timedelta64(purge_days, "D")
        folds.append(
            TemporalFold(
                fold_name="fold_smoke",
                train_idx=np.where(issue_dates < tr_end)[0],
                val_idx=np.where((issue_dates >= v_cut) & (issue_dates < va_end))[0],
                test_idx=np.where(issue_dates >= t_cut)[0],
            )
        )
    return folds