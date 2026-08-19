"""
Meta Continuous Acquisition Blending for batch Bayesian optimization.

A single self-deriving method: instead of choosing one acquisition function per
batch (discrete switching) or combining them with hand-tuned coefficients, it
emits a CONTINUOUS convex blend over UCB / EI / PI whose mixing is governed by
in-campaign signals -- and it DERIVES the influence of each signal from the
campaign's own statistics rather than fixing them by hand.

================================================================================
THE ONE IDEA
================================================================================
Each batch the method produces a convex blend

    score(x) = a_UCB * UCB(x) + a_EI * EI(x) + a_PI * PI(x),   a >= 0, sum = 1

The blend is controlled by a single explore<->exploit weight w in [0,1] (plus an
exploit-pole split between EI and PI). w is computed from statistical signals
read off the campaign itself:

    roughness      short GP lengthscale  -> rugged landscape  -> explore
    uncertainty    high posterior spread -> much still unknown -> explore
    stall          best-so-far flat      -> stuck             -> explore
    dispersion     outcomes collapsed    -> hammering a region -> explore
    improvement    best-so-far rising    -> exploit

What makes it META: the INFLUENCE of each signal on w is not a hand-set constant.
It is estimated from how predictive that signal has been of subsequent
improvement IN THIS CAMPAIGN, then regularized toward a principled prior by
the amount of batch-level evidence accumulated:

    influence(signal) = prior(signal) * (tau + n * gain) / (tau + n)

  * COLD START (n small): influence == prior. With little data the method uses a
    principled, interpretable default -- this IS the cold-start behavior, not a
    separate mode. Reliability is preserved because the method cannot overfit a
    handful of points; it simply uses the prior until evidence arrives.
  * AS EVIDENCE ACCUMULATES (n grows): influence moves toward what the campaign's
    own batch-level statistics support. A 13-batch campaign provides at most
    12 signal-to-next-improvement pairs, so the shrinkage is intentionally conservative.

Strictly self-contained: estimation uses ONLY the current campaign. No
pretraining, no other datasets -- so the method cannot inherit a training
distribution's bias and cannot get stuck because it was tuned on easy data.

================================================================================
COMPONENTS (all part of the one method, not selectable modes)
================================================================================
- Cooling prior on w: governs the early explore->exploit tendency before signals
  are trustworthy; fades as evidence accumulates.
- lambda freedom-dial: w and the EI/PI split define a 1-D "reliable path" through
  the UCB/EI/PI simplex; lambda in [0,1] lets the blend move off that path into
  the simplex interior, and lambda is granted adaptively as the campaign earns it
  (more batches + more available signals -> more freedom). lambda=0 hugs the
  reliable path, lambda->1 uses the full simplex.
- Stall escape: if the campaign is genuinely stuck, w is pushed decisively back
  toward exploration (a strong nudge that grows with how long it has been stuck),
  escalating to a Sobol-reset RECOMMENDATION only if exploring harder still fails.

================================================================================
USING THE OUTPUT
================================================================================
decide_blend(history, ...) returns each batch:
    weights = {"UCB":a, "EI":b, "PI":c}   convex, sum 1   -> score candidates
    w, beta (UCB beta), lambda, p_exploit, signals, meta_influences,
    meta_diagnostics, escape_active, reset_recommended, reason.
Score candidates with the three weights (or evaluate UCB with the returned beta),
take the top-k as the batch, run, append to history, call decide_blend again.

Imports the discrete switcher's validated statistical estimators (MAD noise,
robust slope, adaptive tolerance, dispersion proxy) so signal definitions match
prior work and the switcher remains a clean comparison point.

No dependencies beyond numpy / pandas and the switcher module.
"""

from __future__ import annotations

import importlib as _importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__version__ = "1.1.0"

# v1.1 correctness revisions:
#   - tie-aware rank normalization (flat acquisitions are neutral)
#   - directionally aligned meta-correlation with conservative shrinkage
#   - fixed-unit, nonnegative e-value construction; evidence != confidence
#   - scale-meaningful GP roughness/uncertainty extraction
#   - explicit reset bookkeeping when history contains `reset_executed`

# --------------------------------------------------------------------------- #
# Dependency on the discrete-switcher estimators (lazy + graceful)
# --------------------------------------------------------------------------- #
# The signal estimators (MAD noise, robust slope, adaptive tolerance, dispersion
# proxy) live in the companion "switcher" module so the two methods share
# identical signal definitions. We resolve it LAZILY so that importing this
# module never fails just because the companion file is not yet on the path --
# the error is raised only if/when an estimator is actually needed, with a clear
# message. Set the module via `set_switcher_module(name_or_module)` if your file
# is named differently.
_SWITCHER_CANDIDATE = ("acquisition_switcher",)
_sw = None


def set_switcher_module(module_or_name) -> None:
    """Register the companion switcher module explicitly (module object or name)."""
    global _sw
    if isinstance(module_or_name, str):
        _sw = _importlib.import_module(module_or_name)
    else:
        _sw = module_or_name


def _switcher():
    """Return the switcher module, importing it on first use (cached)."""
    global _sw
    if _sw is not None:
        return _sw
    for _name in _SWITCHER_CANDIDATE:
        try:
            _sw = _importlib.import_module(_name)
            return _sw
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(
        "The companion switcher module (providing the signal estimators) could "
        "not be imported. Place it on your Python path, or call "
        "set_switcher_module(<module or name>). Tried: "
        + ", ".join(_SWITCHER_CANDIDATE)
    )


ACQS: Tuple[str, str, str] = ("UCB", "EI", "PI")
SIGNAL_NAMES: Tuple[str, ...] = (
    "roughness", "uncertainty", "stall", "dispersion", "improvement", "converged",
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class BlendConfig:
    # ---- Cooling prior on w (early explore->exploit tendency) -----------------
    w_start: float = 0.90
    w_floor: float = 0.10
    cooling_k: float = 2.5
    prior_strength: float = 2.0

    # ---- PRIOR signal influences (the meta estimator shrinks toward these) -----
    # These are NOT a separate "fixed mode": they are the principled prior the
    # campaign-derived estimate falls back to when evidence is thin.
    base_w: Dict[str, float] = field(
        default_factory=lambda: {
            "roughness": 0.35,
            "uncertainty": 0.30,
            "stall": 0.30,
            "dispersion": 0.20,
            "improvement": 0.25,
            "converged": 0.0,   # drives the EI->PI split, not w
        }
    )
    sign: Dict[str, float] = field(
        default_factory=lambda: {
            "roughness": +1.0, "uncertainty": +1.0, "stall": +1.0,
            "dispersion": +1.0, "improvement": -1.0, "converged": 0.0,
        }
    )

    # ---- Meta estimation (the data-driven influence derivation) ---------------
    # meta_tau     : prior strength in "effective points"; larger => trust the
    #                prior longer before the data moves an influence.
    # meta_corr_window : trailing batches used for the signal/improvement
    #                association estimate (None = whole campaign).
    # meta_max_gain : cap on how far a data estimate may scale an influence
    #                (x prior), so one spurious correlation cannot dominate.
    meta_tau: float = 4.0
    meta_corr_window: Optional[int] = None
    meta_max_gain: float = 2.0

    # ---- Exploit-pole split: how aggressively to exploit (EI -> PI) -----------
    pi_base: float = 0.0
    pi_converged: float = 0.6
    pi_progress: float = 0.3
    pi_cap: float = 0.8
    min_batches_before_pi: int = 3

    # ---- lambda freedom-dial --------------------------------------------------
    lambda_mode: str = "adaptive"        # "fixed" or "adaptive"
    lambda_value: float = 0.5            # used when lambda_mode="fixed"
    lambda_max: float = 0.8
    lambda_batch_halfsat: float = 6.0
    simplex_temperature: float = 1.0

    # ---- Stall / windows / roughness norm -------------------------------------
    stall_patience: int = 3
    rolling_window: int = 3
    smooth_lengthscale: float = 0.5

    # ---- Stall escape ---------------------------------------------------------
    enable_stall_escape: bool = True
    escape_patience: int = 4
    escape_strength: float = 0.6
    escape_ramp_batches: float = 4.0
    reset_patience: int = 7
    escape_w_floor: float = 0.75

    # ---- Stall-escape runaway guards ------------------------------------------
    # After a reset is recommended, suppress *further* resets for this many
    # batches so the injected diversity points get a chance to move the incumbent
    # before the stall counter (which they don't reset) re-triggers a reset. This
    # breaks the reset-every-round feedback loop.
    reset_cooldown: int = 4
    # Hard cap on resets per campaign. Beyond this, escape may still nudge w but
    # will never re-recommend a reset.
    max_resets: int = 2
    # Grace window: for this many batches after a reset, hold the escape ramp at 0
    # so a single injected-diversity batch cannot immediately re-escalate escape.
    escape_grace: int = 2
    # Deprecated compatibility knob from v1.0; retained so existing config files
    # still load. It is no longer used as a statistical confidence threshold.
    escape_converged_frac: float = 0.98
    # Suppress a disruptive reset when the e-value monitor has strong evidence
    # against the plateau null. This is an evidence threshold, not a posterior probability.
    escape_evalue_p_threshold: float = 0.02
    # Budget-aware escape: once campaign progress exceeds this, do not force
    # random exploration / reset -- there is no budget left to capitalize on new
    # diversity, so exploit instead.
    escape_budget_cutoff: float = 0.80

    # ---- UCB beta mapping -----------------------------------------------------
    beta_min: float = 0.1
    beta_max: float = 3.0

    # ---- Robustness -----------------------------------------------------------
    use_mad_noise: bool = True
    robust: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.w_floor < self.w_start <= 1.0):
            raise ValueError("require 0 <= w_floor < w_start <= 1")
        if self.cooling_k <= 0:
            raise ValueError("cooling_k must be > 0")
        if self.beta_min >= self.beta_max:
            raise ValueError("beta_min must be < beta_max")
        if self.lambda_mode not in ("fixed", "adaptive"):
            raise ValueError("lambda_mode must be 'fixed' or 'adaptive'")
        if not (0.0 <= self.lambda_value <= 1.0):
            raise ValueError("lambda_value must be in [0,1]")
        if not (0.0 <= self.pi_cap <= 1.0):
            raise ValueError("pi_cap must be in [0,1]")
        if self.meta_tau <= 0:
            raise ValueError("meta_tau must be > 0")
        if self.meta_max_gain < 1.0:
            raise ValueError("meta_max_gain must be >= 1")
        if not (0.0 < self.escape_evalue_p_threshold < 1.0):
            raise ValueError("escape_evalue_p_threshold must be in (0,1)")


# --------------------------------------------------------------------------- #
# E-value stopping configuration (SECONDARY, read-only layer)
# --------------------------------------------------------------------------- #
# These knobs control the anytime-valid STOPPING-AND-CONFIDENCE monitor. They do
# NOT affect point selection or the meta blend in any way -- the monitor only
# reads the campaign and reports whether there is enough evidence to stop.
@dataclass
class EValueStopConfig:
    # Significance level for the anytime-valid e-process, conditional on the
    # explicit null assumption described below.
    alpha: float = 0.05

    # FIXED practical-improvement margin in the SAME UNITS as the objective.
    # Examples: use 0.02 if yield is represented on [0,1], or 2.0 if yield is
    # represented as percent yield on [0,100]. Keeping this fixed in advance is
    # essential: scaling it by the observed range would retroactively change old
    # e-variables as new extrema are observed.
    margin: float = 0.02

    # Betting fraction in (0,1). It is mapped to a valid kappa by
    # kappa = bet / (1-p0), guaranteeing e_b >= 0 for X_b in {0,1}.
    bet: float = 0.5

    # Conditional-null assumption: under H0 (not near-best yet), a fresh batch
    # beats the incumbent by > margin with probability at least p0. The formal
    # anytime-valid guarantee is conditional on this assumption being defensible
    # for the adaptive BO policy.
    null_beat_prob: float = 0.5

    # Minimum completed batches before a stop can be recommended.
    min_batches: int = 4

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0,1)")
        if self.margin < 0:
            raise ValueError("margin must be >= 0")
        if not (0.0 < self.bet < 1.0):
            raise ValueError("bet must be in (0,1)")
        if not (0.0 < self.null_beat_prob < 1.0):
            raise ValueError("null_beat_prob must be in (0,1)")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _cooling_prior(progress: float, cfg: BlendConfig) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return cfg.w_floor + (cfg.w_start - cfg.w_floor) * np.exp(-cfg.cooling_k * progress)


def _batches_since_best_improved(best_per_batch: np.ndarray, tol: float) -> int:
    if len(best_per_batch) <= 1:
        return 0
    running = np.maximum.accumulate(best_per_batch)
    stall = 0
    for i in range(len(running) - 1, 0, -1):
        if running[i] - running[i - 1] <= tol:
            stall += 1
        else:
            break
    return stall


def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
    z = np.asarray(x, float) / max(temp, 1e-6)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


# --------------------------------------------------------------------------- #
# Scale-safe blending of acquisition scores (FIX #1)
# --------------------------------------------------------------------------- #
def blend_scores(
    ucb: np.ndarray,
    ei: np.ndarray,
    pi: np.ndarray,
    weights: Dict[str, float],
    *,
    method: str = "rank",
) -> np.ndarray:
    """
    Combine three acquisition-score vectors into ONE blended score per candidate,
    using the convex weights from decide_blend()["weights"].

    WHY THIS EXISTS: UCB, EI and PI live on completely different numeric scales
    (UCB ~ objective scale, EI often ~1e-3 near convergence, PI in [0,1]). A raw
    linear combination is dominated by whichever has the largest magnitude, so
    the weights become meaningless and "blending doesn't work". This helper
    normalizes each acquisition to a COMMON [0,1] scale BEFORE applying weights,
    which is what makes the blend behave as intended.

    Parameters
    ----------
    ucb, ei, pi : array-like, one score per candidate (same length, same order).
    weights     : {"UCB":a, "EI":b, "PI":c}; need not sum to 1 (renormalized here).
    method      : "rank"     -> rank-normalize each to [0,1] (scale-free, robust;
                               recommended, and immune to outliers/units).
                  "minmax"   -> (x - min)/(max - min) per acquisition.
                  "zscore"   -> standardize then squash via logistic.

    Returns
    -------
    np.ndarray of blended scores in [0,1] (higher = more desirable). Use argmax /
    top-k over this to choose the batch.
    """
    u = np.asarray(ucb, float)
    e = np.asarray(ei, float)
    p = np.asarray(pi, float)
    if not (len(u) == len(e) == len(p)):
        raise ValueError("ucb, ei, pi must have the same length")
    n = len(u)
    if n == 0:
        return np.array([], float)

    def _norm(v: np.ndarray) -> np.ndarray:
        if np.all(~np.isfinite(v)):
            return np.zeros_like(v)
        v = np.where(np.isfinite(v), v, np.nanmin(v[np.isfinite(v)]) if np.any(np.isfinite(v)) else 0.0)
        if method == "rank":
            # Tie-aware average ranks. The previous double-argsort assigned
            # different scores to exactly tied candidates purely from array order.
            if n <= 1:
                return np.zeros_like(v)
            order = np.argsort(v, kind="mergesort")
            sorted_v = v[order]
            ranks = np.empty(n, dtype=float)
            i = 0
            while i < n:
                j = i + 1
                while j < n and sorted_v[j] == sorted_v[i]:
                    j += 1
                avg_rank = 0.5 * (i + (j - 1))
                ranks[order[i:j]] = avg_rank
                i = j
            # A fully flat acquisition should be neutral, not arbitrarily ordered.
            if np.all(ranks == ranks[0]):
                return np.full_like(v, 0.5, dtype=float)
            return ranks / (n - 1)
        if method == "minmax":
            lo, hi = v.min(), v.max()
            return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
        if method == "zscore":
            sd = v.std()
            z = (v - v.mean()) / sd if sd > 1e-12 else np.zeros_like(v)
            return 1.0 / (1.0 + np.exp(-z))
        raise ValueError("method must be 'rank', 'minmax', or 'zscore'")

    wsum = weights.get("UCB", 0) + weights.get("EI", 0) + weights.get("PI", 0)
    if wsum <= 0:
        wsum = 1.0
    a = weights.get("UCB", 0) / wsum
    b = weights.get("EI", 0) / wsum
    c = weights.get("PI", 0) / wsum
    return a * _norm(u) + b * _norm(e) + c * _norm(p)


# --------------------------------------------------------------------------- #
# Signal extraction (current batch)
# --------------------------------------------------------------------------- #
def _extract_signals(
    history, bs, cfg, sw_cfg, batch_col, yield_col, yrange,
    lengthscale_norm, mean_uncertainty, diversity,
) -> Dict[str, Optional[float]]:
    try:
        tol = _switcher()._adaptive_yield_tolerance(bs, sw_cfg, yrange, "delta_batch_average")
    except Exception:
        tol = 0.02 * yrange

    roughness = None
    if lengthscale_norm is not None:
        ls = float(np.clip(lengthscale_norm, 0.0, 1.0))
        roughness = float(np.clip((cfg.smooth_lengthscale - ls) / cfg.smooth_lengthscale, 0.0, 1.0))

    unc = float(np.clip(mean_uncertainty, 0.0, 1.0)) if mean_uncertainty is not None else None

    if "best_yield_so_far" in bs:
        best_per_batch = bs["best_yield_so_far"].to_numpy(float)
    elif "batch_best_yield" in bs:
        best_per_batch = np.maximum.accumulate(bs["batch_best_yield"].to_numpy(float))
    else:
        best_per_batch = np.maximum.accumulate(bs["batch_average_yield"].to_numpy(float))
    stall_batches = _batches_since_best_improved(best_per_batch, tol)
    stall = float(np.clip(stall_batches / max(cfg.stall_patience, 1), 0.0, 1.0))

    dispersion = None
    if diversity is not None:
        dispersion = 1.0 - float(np.clip(diversity, 0.0, 1.0))
    else:
        try:
            disp_ratio, _drift = _switcher()._dispersion_and_drift(
                history, bs, batch_col, yield_col, yrange, sw_cfg.rolling_window
            )
            if disp_ratio is not None:
                dispersion = float(np.clip(1.0 - disp_ratio, 0.0, 1.0))
        except Exception:
            dispersion = None

    try:
        slope = _switcher()._rolling_slope(bs["delta_batch_average"].to_numpy(float)[-sw_cfg.rolling_window:])
    except Exception:
        slope = 0.0
    improving = float(np.clip(slope / (tol + 1e-12), 0.0, 1.0))

    if unc is not None:
        converged = float(np.clip(1.0 - unc, 0.0, 1.0))
    else:
        converged = float(np.clip(stall, 0.0, 1.0)) * 0.5

    return {
        "roughness": roughness, "uncertainty": unc, "stall": stall,
        "dispersion": dispersion, "improvement": improving, "converged": converged,
        "_tol": tol, "_stall_batches": float(stall_batches),
        "_best": float(best_per_batch[-1]) if len(best_per_batch) else None,
        "_best_per_batch": best_per_batch,
    }


# --------------------------------------------------------------------------- #
# Meta estimation: derive signal influences from in-campaign statistics
# --------------------------------------------------------------------------- #
def _recompute_signal_history(bs, cfg, sw_cfg, yrange):
    """Rebuild per-batch values of the data-derivable signals and the realized
    improvement that followed each batch, so the meta estimator can relate them.
    Model signals (roughness, uncertainty) are not stored historically and keep
    their prior influence; the data signals (stall, improvement, dispersion)
    carry the estimate, reconstructed per-batch from the yield record."""
    n = len(bs)
    if "best_yield_so_far" in bs:
        best = bs["best_yield_so_far"].to_numpy(float)
    elif "batch_best_yield" in bs:
        best = np.maximum.accumulate(bs["batch_best_yield"].to_numpy(float))
    else:
        best = np.maximum.accumulate(bs["batch_average_yield"].to_numpy(float))

    try:
        tol = _switcher()._adaptive_yield_tolerance(bs, sw_cfg, yrange, "delta_batch_average")
    except Exception:
        tol = 0.02 * yrange

    improve = np.full(n, np.nan)
    improve[:-1] = np.diff(best)   # improvement AFTER each batch

    stall_hist = np.zeros(n)
    for k in range(n):
        stall_hist[k] = _batches_since_best_improved(best[: k + 1], tol) / max(cfg.stall_patience, 1)
    stall_hist = np.clip(stall_hist, 0.0, 1.0)

    deltas = bs["delta_batch_average"].to_numpy(float)
    impr_hist = np.zeros(n)
    for k in range(n):
        w0 = max(0, k - sw_cfg.rolling_window + 1)
        try:
            slope = _switcher()._rolling_slope(deltas[w0 : k + 1])
        except Exception:
            slope = 0.0
        impr_hist[k] = np.clip(slope / (tol + 1e-12), 0.0, 1.0)

    # FIX #3: reconstruct a real per-batch dispersion history instead of NaN.
    # dispersion signal = collapse of within-batch yield spread relative to the
    # campaign's typical spread. Low recent spread (outcomes bunched up) -> high
    # dispersion-collapse signal -> historically associatable with improvement.
    disp_hist = np.full(n, np.nan)
    if "batch_yield_std" in bs:
        spread = bs["batch_yield_std"].to_numpy(float)
    elif "batch_yield_iqr" in bs:
        spread = bs["batch_yield_iqr"].to_numpy(float)
    else:
        spread = None
    if spread is not None and np.any(np.isfinite(spread)):
        ref = np.nanmedian(spread[np.isfinite(spread)])
        ref = ref if (ref is not None and ref > 1e-12) else 1.0
        # 1 - (spread/ref) clipped: small spread -> near 1 (collapsed)
        disp_hist = np.clip(1.0 - (spread / ref), 0.0, 1.0)

    return {"stall": stall_hist, "improvement": impr_hist, "dispersion": disp_hist}, improve


def _meta_influences(bs, cfg, sw_cfg, yrange, base_w):
    """Regularized empirical estimate of each signal's influence on w.

    For exploration-driving signals (stall/dispersion), usefulness means that a
    HIGH signal tends to precede LOW subsequent improvement. For the exploitation-
    driving improvement signal, usefulness means HIGH signal tends to precede HIGH
    subsequent improvement. We therefore orient each raw Pearson correlation by
    the intended signal direction before converting it to a bounded gain.
    """
    hist, improve = _recompute_signal_history(bs, cfg, sw_cfg, yrange)
    eff = dict(base_w)
    diagnostics: Dict[str, Any] = {}
    win = cfg.meta_corr_window
    for s, series in hist.items():
        prior = base_w.get(s, 0.0)
        if prior <= 0:
            continue
        x = series.astype(float)
        y = improve.astype(float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if win is not None:
            idx = np.where(mask)[0]
            if len(idx) > win:
                keep = idx[-win:]
                m2 = np.zeros_like(mask)
                m2[keep] = True
                mask = mask & m2
        xv, yv = x[mask], y[mask]
        n = len(xv)
        if n < 3 or np.std(xv) < 1e-9 or np.std(yv) < 1e-9:
            diagnostics[s] = {"n": int(n), "gain": 1.0, "corr": None, "oriented_corr": None}
            continue

        corr = float(np.corrcoef(xv, yv)[0, 1])
        corr = 0.0 if np.isnan(corr) else float(np.clip(corr, -1.0, 1.0))
        # sign=+1 means the signal calls for exploration, so anti-correlation
        # with future improvement is evidence that the signal is informative.
        # sign=-1 (improvement) means positive correlation is informative.
        oriented_corr = float(np.clip(-cfg.sign.get(s, 0.0) * corr, -1.0, 1.0))
        gain = float(np.exp(oriented_corr * np.log(cfg.meta_max_gain)))
        shrunk_gain = (cfg.meta_tau + n * gain) / (cfg.meta_tau + n)
        eff[s] = prior * shrunk_gain
        diagnostics[s] = {
            "n": int(n),
            "gain": round(gain, 3),
            "corr": round(corr, 3),
            "oriented_corr": round(oriented_corr, 3),
            "eff": round(eff[s], 3),
        }
    return eff, diagnostics


# --------------------------------------------------------------------------- #
# Weight construction
# --------------------------------------------------------------------------- #
def _signal_nudge(
    signals: Dict[str, Any],
    eff_w: Dict[str, float],
    cfg: "BlendConfig",
) -> Tuple[float, Dict[str, float]]:
    """Sum the (effective-influence x signal-value x sign) contributions that
    push the explore<->exploit weight. Returns (nudge, per-signal contributions)."""
    nudge = 0.0
    contributions: Dict[str, float] = {}
    for s in ("roughness", "uncertainty", "stall", "dispersion", "improvement"):
        val = signals.get(s)
        if val is None:
            continue
        signed = eff_w.get(s, cfg.base_w[s]) * val * cfg.sign[s]
        nudge += signed
        contributions[s] = signed
    return nudge, contributions


def _explore_weight(
    w_prior: float, nudge: float, n_batches: int, cfg: "BlendConfig"
) -> float:
    """Blend the cooling-prior weight with the signal-driven target. The prior
    dominates early (few batches) and fades as evidence accumulates."""
    n_eff = max(n_batches - 1, 0)
    prior_pull = cfg.prior_strength / (cfg.prior_strength + n_eff)
    w_signal = float(np.clip(w_prior + (2.0 * _sigmoid(nudge) - 1.0), 0.0, 1.0))
    return float(np.clip(prior_pull * w_prior + (1.0 - prior_pull) * w_signal, 0.0, 1.0))


@dataclass
class _EscapeState:
    active: bool = False
    ramp: float = 0.0
    reset_recommended: bool = False
    w_before: float = 0.0


def _reset_history(best_per_batch: np.ndarray, tol: float, cfg: "BlendConfig") -> Tuple[int, int]:
    """Reconstruct escape/reset history from the incumbent trajectory alone, so
    that decide_blend can stay a pure function of `history` while still enforcing
    a reset cooldown and a per-campaign reset cap.

    Replays the stall counter forward over the campaign. Each batch where the
    (cooldown-and-cap-gated) reset condition would have fired counts as a reset;
    a reset zeroes the effective stall counter (the grace period). Returns
    (n_resets_so_far, batches_since_last_reset). This mirrors, in reconstruction,
    exactly the guarded rule applied live below.
    """
    running = np.maximum.accumulate(best_per_batch) if len(best_per_batch) else best_per_batch
    n = len(running)
    if n <= 1:
        return 0, 10**9
    # Replay the guarded reset rule over batches STRICTLY BEFORE the current
    # decision point (indices 1..n-2). The decision for the current batch (n-1)
    # is left to the live _apply_stall_escape call, so we don't double-count the
    # very reset we're about to (maybe) emit. Returns the reset count and the
    # gap since the last reset, both as of just before the current batch.
    n_resets = 0
    last_reset_at = -(10**9)
    eff_stall = 0
    for i in range(1, n - 1):
        improved = (running[i] - running[i - 1]) > tol
        eff_stall = 0 if improved else eff_stall + 1
        since_reset = i - last_reset_at
        if (
            n_resets < cfg.max_resets
            and since_reset > cfg.reset_cooldown
            and eff_stall >= cfg.reset_patience
        ):
            n_resets += 1
            last_reset_at = i
            eff_stall = 0  # grace: injected diversity gets a clean slate
    batches_since_reset = (n - 1 - last_reset_at) if last_reset_at > -(10**9) else 10**9
    return n_resets, batches_since_reset


def _apply_stall_escape(
    w: float,
    stall_batches: int,
    cfg: "BlendConfig",
    *,
    incumbent_frac: float = 0.0,   # retained for API compat; gating now in decide_blend
    progress: float = 0.0,         # retained for API compat; budget gate in decide_blend
    n_resets: int = 0,
    batches_since_reset: int = 10**9,
) -> Tuple[float, _EscapeState]:
    """Push w back toward exploration when genuinely stuck -- with runaway guards
    so escape cannot re-trigger itself every round.

    The mild w-nudge still fires whenever stalled (cheap, helps genuinely stuck
    runs). The disruptive Sobol-RESET recommendation is bounded here by:
      * Grace window: for escape_grace batches after a reset, hold the ramp at 0
        so a single injected-diversity batch cannot instantly re-escalate.
      * Reset cooldown + per-campaign cap: only recommend a reset if enough
        batches have passed since the last one AND the cap is not yet hit.
    The convergence gate (near-best) and budget-tail gate on the RESET flag are
    applied in decide_blend, where the e-value monitor's evidence is available.
    """
    st = _EscapeState(w_before=w)
    if not (cfg.enable_stall_escape and stall_batches >= cfg.escape_patience):
        return w, st

    st.active = True
    # Grace: freeze the ramp right after a reset so injected points can land.
    if batches_since_reset <= cfg.escape_grace:
        st.ramp = 0.0
    else:
        st.ramp = float(np.clip(
            (stall_batches - cfg.escape_patience) / max(cfg.escape_ramp_batches, 1e-9),
            0.0, 1.0,
        ))
    w = w + cfg.escape_strength * st.ramp * (1.0 - w)
    w = max(w, cfg.escape_w_floor * st.ramp + st.w_before * (1.0 - st.ramp))
    w = float(np.clip(w, 0.0, 1.0))

    # Reset recommendation, now cooldown- and cap-gated (hard runaway bound).
    st.reset_recommended = (
        stall_batches >= cfg.reset_patience
        and n_resets < cfg.max_resets
        and batches_since_reset > cfg.reset_cooldown
    )
    return w, st


def _merge_simplex(
    w: float, p: float, n_batches: int, signals: Dict[str, Any], cfg: "BlendConfig"
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], float]:
    """Build the A-path and B-simplex points, interpolate by lambda, renormalize.
    Returns (weights, A_point, B_point, lambda)."""
    a_pt = _A_path_point(w, p)
    b_pt = _B_simplex_point(w, p, cfg)
    lam = _resolve_lambda(cfg, n_batches, signals)
    weights = {k: (1.0 - lam) * a_pt[k] + lam * b_pt[k] for k in ACQS}
    tot = sum(weights.values()) or 1.0
    weights = {k: v / tot for k, v in weights.items()}
    return weights, a_pt, b_pt, lam


def _format_reason(
    w: float, p: float, lam: float, weights: Dict[str, float],
    contributions: Dict[str, float], stall_batches: int, escape: _EscapeState,
) -> str:
    """Human-readable, interpretable one-line summary of the decision."""
    drivers = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))
    top = ", ".join(f"{k}{'+' if v >= 0 else ''}{v:.2f}" for k, v in drivers[:3])
    reason = (
        f"[meta] w={w:.2f} p={p:.2f} lam={lam:.2f} | "
        f"UCB/EI/PI={weights['UCB']:.2f}/{weights['EI']:.2f}/{weights['PI']:.2f} | "
        f"top: {top or 'none'}; stall={stall_batches}b"
    )
    if escape.active:
        reason += (
            f"; ESCAPE(ramp={escape.ramp:.2f}, w {escape.w_before:.2f}->{w:.2f}"
            f"{', RESET' if escape.reset_recommended else ''})"
        )
    return reason


def _exploit_split(signals, cfg, n_batches) -> float:
    if n_batches < cfg.min_batches_before_pi:
        return 0.0
    conv = signals.get("converged") or 0.0
    impr = signals.get("improvement") or 0.0
    p = cfg.pi_base + cfg.pi_converged * conv + cfg.pi_progress * impr
    return float(np.clip(p, 0.0, cfg.pi_cap))


def _A_path_point(w: float, p: float) -> Dict[str, float]:
    exploit = 1.0 - w
    return {"UCB": w, "EI": exploit * (1.0 - p), "PI": exploit * p}


def _B_simplex_point(w: float, p: float, cfg: BlendConfig) -> Dict[str, float]:
    d_ucb = 2.0 * w
    d_ei = 2.0 * (1.0 - w) * (1.0 - p) + 0.25
    d_pi = 2.0 * (1.0 - w) * p + 0.25
    a = _softmax(np.array([d_ucb, d_ei, d_pi]), cfg.simplex_temperature)
    return {"UCB": float(a[0]), "EI": float(a[1]), "PI": float(a[2])}


def _resolve_lambda(cfg: BlendConfig, n_batches: int, signals: Dict[str, Any]) -> float:
    if cfg.lambda_mode == "fixed":
        return float(cfg.lambda_value)
    batch_term = n_batches / (n_batches + cfg.lambda_batch_halfsat)
    present = sum(1 for s in ("roughness", "uncertainty", "stall", "dispersion")
                  if signals.get(s) is not None)
    cond_term = present / 4.0
    lam = cfg.lambda_max * batch_term * (0.5 + 0.5 * cond_term)
    return float(np.clip(lam, 0.0, cfg.lambda_max))


# --------------------------------------------------------------------------- #
# E-value stopping monitor (SECONDARY, read-only)
# --------------------------------------------------------------------------- #
# The algebra below is an anytime-valid e-process PROVIDED the conditional-null
# assumption E[X_b | past] >= p0 is valid for the adaptive BO policy. This code
# therefore reports an anytime p-value/evidence score, not a posterior probability
# that the incumbent is near-best.
def _evalue_stop(history, bs, batch_col, yield_col, yrange, ecfg) -> Dict[str, Any]:
    n_batches = len(bs)
    if "best_yield_so_far" in bs:
        best = bs["best_yield_so_far"].to_numpy(float)
    elif "batch_best_yield" in bs:
        best = np.maximum.accumulate(bs["batch_best_yield"].to_numpy(float))
    else:
        best = np.maximum.accumulate(bs["batch_average_yield"].to_numpy(float))

    if "batch_best_yield" in bs:
        batch_max = bs["batch_best_yield"].to_numpy(float)
    else:
        batch_max = bs.groupby(batch_col)[yield_col].max().to_numpy(float) \
            if batch_col in bs else best

    # IMPORTANT: fixed in objective units. Do NOT rescale by the observed range.
    margin_abs = float(ecfg.margin)
    p0 = float(np.clip(ecfg.null_beat_prob, 1e-6, 1.0 - 1e-6))

    # e_b = 1 + kappa(p0-X_b). For X_b=1, nonnegativity requires
    # kappa <= 1/(1-p0). Mapping bet in (0,1) as below guarantees this.
    kappa = float(ecfg.bet / (1.0 - p0))

    e_process = 1.0
    e_log = []
    for b in range(1, n_batches):
        incumbent = best[b - 1]
        x_b = 1.0 if (batch_max[b] - incumbent) > margin_abs else 0.0
        e_b = 1.0 + kappa * (p0 - x_b)
        # Numerical guard only; analytical construction is nonnegative.
        if e_b < -1e-12:
            raise RuntimeError("invalid negative e-variable; check EValueStopConfig")
        e_b = max(e_b, 0.0)
        e_process *= e_b
        e_log.append(e_process)

    threshold = 1.0 / ecfg.alpha
    can_stop = n_batches >= ecfg.min_batches
    stop_recommended = bool(can_stop and e_process >= threshold)

    running_max = max(e_log) if e_log else 1.0
    anytime_p = float(min(1.0, 1.0 / running_max)) if running_max > 0 else 1.0
    # Convenience monotone visualization only; NOT statistical confidence.
    evidence_score = float(np.clip(1.0 - anytime_p, 0.0, 1.0))

    return {
        "e_process": float(e_process),
        "threshold": float(threshold),
        "stop_recommended": stop_recommended,
        "anytime_p_value": anytime_p,
        "evidence_score": evidence_score,
        # Backward-compatible alias. Do not interpret as posterior confidence.
        "confidence_near_best": evidence_score,
        "margin_abs": margin_abs,
        "null_beat_prob": p0,
        "kappa": kappa,
        "validity_note": (
            "Anytime validity is conditional on E[X_b|past] >= null_beat_prob "
            "under the adaptive sampling policy."
        ),
        "incumbent": float(best[-1]) if len(best) else None,
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def decide_blend(
    history: pd.DataFrame,
    *,
    batch_col: str = "batch",
    yield_col: str = "yield",
    acq_col: str = "acquisition",
    reset_col: str = "reset_executed",
    batches_remaining: Optional[int] = None,
    total_batches: Optional[int] = None,
    lengthscale_norm: Optional[float] = None,
    mean_uncertainty: Optional[float] = None,
    diversity: Optional[float] = None,
    config: Optional[BlendConfig] = None,
    evalue_stop: Optional[EValueStopConfig] = None,
) -> Dict[str, Any]:
    """
    Compute the meta continuous 3-way acquisition blend for the next batch.

    Signal influences are derived from in-campaign statistics (shrinking toward a
    principled prior); with little data the method uses the prior (its cold-start
    behavior) and adapts as evidence accumulates.

    Returns dict:
      weights        : {"UCB":a,"EI":b,"PI":c}  convex, sum 1   (score with these)
      w              : explore<->exploit summary in [0,1]
      beta           : UCB beta implied by w
      lambda         : freedom used this batch (0=reliable path, 1=full simplex)
      p_exploit      : EI->PI split used
      meta_influences: the data-derived per-signal influences this batch
      meta_diagnostics: per-signal {n, corr, oriented_corr, gain, eff}
      escape_active, escape_ramp, reset_recommended : stall-escape state
      A_point, B_point, signals, reason, progress, batch_summary
    """
    cfg = config or BlendConfig()

    # FIX #4: defensive guards for tiny / degenerate campaigns. With no data, or
    # before any batch has completed, return a principled cold-start blend (pure
    # cooling prior at maximum exploration) rather than touching estimators that
    # assume >=1 completed batch or a positive yield range.
    if history is None or len(history) == 0 or yield_col not in getattr(history, "columns", []):
        w0 = cfg.w_start
        beta0 = cfg.beta_min + (cfg.beta_max - cfg.beta_min) * w0
        out = {
            "weights": {"UCB": w0, "EI": 1.0 - w0, "PI": 0.0},
            "w": w0, "beta": beta0, "lambda": 0.0, "p_exploit": 0.0,
            "meta_influences": {k: round(v, 3) for k, v in cfg.base_w.items()},
            "meta_diagnostics": {}, "A_point": {"UCB": w0, "EI": 1 - w0, "PI": 0.0},
            "B_point": {"UCB": w0, "EI": 1 - w0, "PI": 0.0},
            "signals": {}, "stall_batches": 0, "escape_active": False,
            "escape_ramp": 0.0, "reset_recommended": False, "tolerance": None,
            "reason": "[meta] cold start (no completed data): full exploration prior",
            "progress": 0.0, "batch_summary": None,
        }
        if evalue_stop is not None:
            out["evalue"] = {"e_process": 1.0, "threshold": 1.0 / evalue_stop.alpha,
                             "stop_recommended": False, "anytime_p_value": 1.0,
                             "evidence_score": 0.0, "confidence_near_best": 0.0, "incumbent": None}
        return out

    sw_cfg = _switcher().SwitchConfig(
        use_mad_noise=cfg.use_mad_noise, robust=cfg.robust, rolling_window=cfg.rolling_window
    )

    bs = _switcher().summarize_batches(history, batch_col, yield_col, acq_col)
    n_batches = len(bs)

    yvals = pd.to_numeric(history[yield_col], errors="coerce").dropna().to_numpy(float)
    yrange = float(yvals.max() - yvals.min()) if len(yvals) else 1.0
    yrange = yrange if yrange > 0 else 1.0

    if total_batches and total_batches > 0:
        progress = n_batches / float(total_batches)
    elif batches_remaining is not None:
        progress = n_batches / float(n_batches + max(batches_remaining, 0) + 1e-9)
    else:
        progress = 1.0 - np.exp(-n_batches / 8.0)
    w_prior = _cooling_prior(progress, cfg)

    signals = _extract_signals(
        history, bs, cfg, sw_cfg, batch_col, yield_col, yrange,
        lengthscale_norm, mean_uncertainty, diversity,
    )

    # ---- META: derive per-signal influences from in-campaign statistics ------
    eff_w, meta_diag = _meta_influences(bs, cfg, sw_cfg, yrange, cfg.base_w)

    # ---- explore<->exploit weight w ------------------------------------------
    nudge, contributions = _signal_nudge(signals, eff_w, cfg)
    w = _explore_weight(w_prior, nudge, n_batches, cfg)

    # ---- stall escape: push w back up when genuinely stuck -------------------
    stall_batches_n = int(signals["_stall_batches"])
    # Guard inputs: how close the incumbent is to the best-seen yield (a stall
    # here == convergence), how much budget is left, and the reconstructed reset
    # history (cooldown + per-campaign cap) so escape cannot re-fire every round.
    best_per_batch = signals.get("_best_per_batch")
    tol_n = float(signals.get("_tol") or (0.02 * yrange))
    if reset_col in history.columns and batch_col in history.columns:
        # Prefer explicit executed-reset bookkeeping. This avoids treating an
        # ignored recommendation as though a reset really happened.
        reset_raw = history[reset_col]
        if pd.api.types.is_bool_dtype(reset_raw):
            reset_mask = reset_raw.fillna(False)
        else:
            reset_mask = (
                reset_raw.fillna("").astype(str).str.strip().str.lower()
                .isin({"1", "true", "yes", "y", "reset", "executed"})
            )
        reset_batches = pd.to_numeric(history.loc[reset_mask, batch_col], errors="coerce").dropna()
        reset_batches = sorted(set(int(x) for x in reset_batches))
        current_batch_index = n_batches - 1
        prior_reset_batches = [b for b in reset_batches if b < current_batch_index]
        n_resets_prior = len(prior_reset_batches)
        batches_since_reset = (
            current_batch_index - prior_reset_batches[-1] if prior_reset_batches else 10**9
        )
    elif best_per_batch is not None and len(best_per_batch):
        # Backward-compatible fallback for histories that do not record resets.
        n_resets_prior, batches_since_reset = _reset_history(best_per_batch, tol_n, cfg)
    else:
        n_resets_prior, batches_since_reset = 0, 10**9
    # Convergence signal: prefer the anytime-valid e-value monitor's evidence
    # that the incumbent is near-best (computed below); it is the principled
    # "are we converged" test this module already provides. We read it after the
    # e-value block, so escape is applied there. For now compute w-escape without
    # the convergence gate; the reset cooldown + cap + grace already bound any
    # runaway, and the convergence/budget gates are applied to the *reset flag*
    # just below once e-value evidence is known.
    w, escape = _apply_stall_escape(
        w, stall_batches_n, cfg,
        incumbent_frac=0.0,
        progress=progress,
        n_resets=n_resets_prior,
        batches_since_reset=batches_since_reset,
    )

    # ---- exploit split p (EI -> PI); suppressed while escaping ----------------
    p = _exploit_split(signals, cfg, n_batches)
    if escape.active:
        p = p * (1.0 - escape.ramp)

    # ---- simplex points and lambda merge -------------------------------------
    weights, A_pt, B_pt, lam = _merge_simplex(w, p, n_batches, signals, cfg)
    beta = cfg.beta_min + (cfg.beta_max - cfg.beta_min) * w

    reason = _format_reason(w, p, lam, weights, contributions, stall_batches_n, escape)

    out = {
        "weights": weights,
        "w": w,
        "beta": beta,
        "lambda": lam,
        "p_exploit": p,
        "meta_influences": {k: round(v, 3) for k, v in eff_w.items()},
        "meta_diagnostics": meta_diag,
        "A_point": A_pt,
        "B_point": B_pt,
        "signals": {k: v for k, v in signals.items() if not k.startswith("_")},
        "stall_batches": stall_batches_n,
        "escape_active": escape.active,
        "escape_ramp": escape.ramp,
        "reset_recommended": escape.reset_recommended,
        "tolerance": signals["_tol"],
        "reason": reason,
        "progress": progress,
        "batch_summary": bs,
    }

    # ---- SECONDARY: anytime-valid e-value stopping monitor (read-only) -------
    # Runs only if an EValueStopConfig is supplied. Does NOT affect anything
    # above -- it only adds a stopping-and-evidence report.
    if evalue_stop is not None:
        ev = _evalue_stop(history, bs, batch_col, yield_col, yrange, evalue_stop)
        out["evalue"] = ev
        out["reason"] += (
            f" | E={ev['e_process']:.2f}/{ev['threshold']:.1f}"
            f" p={ev['anytime_p_value']:.3f}"
            f"{' STOP' if ev['stop_recommended'] else ''}"
        )

        # ---- Evidence gate on RESET ------------------------------------------
        # Use the anytime p-value directly. It is an evidence measure, not a
        # probability that the incumbent is near-best.
        anytime_p = float(ev.get("anytime_p_value", 1.0))
        if out["reset_recommended"] and (
            ev.get("stop_recommended") or anytime_p <= cfg.escape_evalue_p_threshold
        ):
            out["reset_recommended"] = False
            out["reason"] += " | RESET suppressed (strong plateau evidence)"

    # ---- Budget gate on RESET (applies with or without the e-value monitor) ---
    # Past the budget cutoff there is no runway left to exploit fresh diversity,
    # so never recommend a disruptive reset in the tail of the campaign.
    if out["reset_recommended"] and progress >= cfg.escape_budget_cutoff:
        out["reset_recommended"] = False
        out["reason"] += " | RESET suppressed (budget tail)"

    return out


__all__ = ["BlendConfig", "EValueStopConfig", "decide_blend", "blend_scores",
           "extract_gp_signals", "set_switcher_module", "ACQS", "SIGNAL_NAMES",
           "__version__"]


# --------------------------------------------------------------------------- #
# GP signal extractor for BayBE / BoTorch surrogates (FIX #5)
# --------------------------------------------------------------------------- #
def extract_gp_signals(
    surrogate, candidates=None, *, search_ranges=None, uncertainty_reference=None
):
    """Extract scale-meaningful GP roughness and uncertainty signals.

    Parameters
    ----------
    surrogate : fitted BayBE/BoTorch/gpytorch surrogate.
    candidates : optional candidate tensor/array used for posterior uncertainty.
    search_ranges : per-dimension physical/search-box ranges. REQUIRED for a
        trustworthy normalized lengthscale. If unavailable, lengthscale_norm is
        returned as None rather than normalizing lengthscales by themselves.
    uncertainty_reference : optional positive objective-scale reference for
        posterior standard deviation. If omitted, the function tries to infer a
        scale from model training targets. If neither is available, uncertainty
        is returned as None rather than mean(std)/max(std), which confounds
        relative uniformity with absolute uncertainty.

    Returns
    -------
    (lengthscale_norm, mean_uncertainty), each in [0,1] or None.
    """
    import numpy as _np

    def _to_numpy(t):
        try:
            return t.detach().cpu().numpy()
        except Exception:
            return _np.asarray(t)

    def _positive_scale(values):
        try:
            y = _to_numpy(values).astype(float).reshape(-1)
            y = y[_np.isfinite(y)]
        except Exception:
            return None
        if y.size < 2:
            return None
        q25, q75 = _np.percentile(y, [25, 75])
        iqr = float(q75 - q25)
        sd = float(_np.std(y))
        rng = float(_np.max(y) - _np.min(y))
        # Prefer a robust scale but fall back gracefully for tiny/degenerate sets.
        for val in (iqr / 1.349 if iqr > 0 else 0.0, sd, rng):
            if _np.isfinite(val) and val > 1e-12:
                return float(val)
        return None

    model = None
    for path in ("_model", "model", "gp", "_gp"):
        m = getattr(surrogate, path, None)
        if m is not None:
            model = m
            break
    if model is None:
        model = surrogate

    # --- normalized lengthscale ---------------------------------------------
    lengthscale_norm = None
    ls = None
    for path in ("covar_module.base_kernel.lengthscale",
                 "covar_module.lengthscale",
                 "kernel.lengthscale"):
        obj = model
        ok = True
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok:
            ls = _to_numpy(obj).reshape(-1)
            break

    if ls is not None and ls.size > 0 and search_ranges is not None:
        rng = _np.asarray(search_ranges, float).reshape(-1)
        if rng.size == ls.size and _np.all(_np.isfinite(rng)) and _np.all(rng > 0):
            lsn = ls / rng
            lengthscale_norm = float(_np.clip(_np.mean(lsn), 0.0, 1.0))

    # --- absolute posterior uncertainty ------------------------------------
    mean_uncertainty = None
    if candidates is not None:
        try:
            try:
                import torch
                X = candidates
                if not hasattr(X, "dtype") or "tensor" not in str(type(X)).lower():
                    X = torch.as_tensor(_np.asarray(candidates, float))
                model.eval()
                with torch.no_grad():
                    post = model.posterior(X) if hasattr(model, "posterior") else model(X)
                    if hasattr(post, "variance"):
                        std = post.variance.sqrt()
                    else:
                        std = post.stddev
                std = _to_numpy(std).astype(float).reshape(-1)
            except Exception:
                pred = surrogate.posterior(candidates)
                if hasattr(pred, "stddev"):
                    std = _to_numpy(pred.stddev).astype(float).reshape(-1)
                elif hasattr(pred, "variance"):
                    std = _np.sqrt(_to_numpy(pred.variance).astype(float)).reshape(-1)
                else:
                    raise AttributeError("posterior has neither stddev nor variance")

            std = std[_np.isfinite(std)]
            if std.size:
                ref = float(uncertainty_reference) if uncertainty_reference is not None else None
                if ref is not None and (not _np.isfinite(ref) or ref <= 0):
                    raise ValueError("uncertainty_reference must be positive and finite")
                if ref is None:
                    for attr in ("train_targets", "train_y", "y_train_"):
                        target = getattr(model, attr, None)
                        if target is not None:
                            ref = _positive_scale(target)
                            if ref is not None:
                                break
                if ref is not None and ref > 0:
                    mean_uncertainty = float(_np.clip(_np.mean(std) / ref, 0.0, 1.0))
        except Exception:
            mean_uncertainty = None

    return lengthscale_norm, mean_uncertainty


# --------------------------------------------------------------------------- #
# Self-test / usage example
# --------------------------------------------------------------------------- #
def _smoke_test() -> None:
    """Minimal self-verifying example. Run `python meta_acquisition_blender_evalue.py`.
    Demonstrates a full 13-batch campaign loop and checks core invariants."""
    rng = np.random.default_rng(0)
    rows: List[dict] = []
    total = 13
    for b in range(total):
        base = min(0.95, 0.30 + 0.05 * b)  # smooth converging landscape
        for _ in range(6):
            rows.append({
                "batch": b,
                "yield": float(np.clip(base + rng.normal(0, 0.05), 0.0, 1.0)),
                "acquisition": "blend",
            })
        history = pd.DataFrame(rows)
        out = decide_blend(
            history,
            total_batches=total,
            lengthscale_norm=0.6,
            mean_uncertainty=max(0.1, 0.8 - 0.05 * b),
            evalue_stop=EValueStopConfig(),
        )
        w = out["weights"]
        assert abs(sum(w.values()) - 1.0) < 1e-9, "weights must sum to 1"
        assert all(v >= -1e-12 for v in w.values()), "weights must be nonnegative"
        assert 0.0 <= out["w"] <= 1.0, "w must be in [0,1]"

    # candidate scoring with the returned weights
    ucb = rng.normal(size=20) * 50          # large scale
    ei = rng.uniform(1e-4, 1e-3, size=20)   # tiny scale
    pi = rng.uniform(0, 1, size=20)         # mid scale
    blended = blend_scores(ucb, ei, pi, out["weights"])
    assert blended.shape == (20,)
    assert float(blended.min()) >= 0.0 and float(blended.max()) <= 1.0

    print(f"meta_acquisition_blender_evalue v{__version__}: smoke test passed.")
    print("final reason:", out["reason"])


if __name__ == "__main__":
    _smoke_test()
