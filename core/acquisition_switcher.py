"""
Adaptive acquisition-function switching for batch Bayesian optimization (v4).

Decides, after every completed batch, which acquisition function (UCB, EI, PI)
the optimizer should use for the next batch, based only on observed data plus
optional normalized model diagnostics.

Key design goals
----------------
1. Scale-invariant: works for percent yields (0-100), fractional conversion
   (0-1), ee, or any other maximization objective, including negative-valued
   ones. All thresholds derive from the observed data range and noise unless
   you opt out (``auto_scale=False``).
2. Noise-aware: "improvement" and "stagnation" are judged against an adaptive
   tolerance estimated from recent batch-to-batch variation.
3. Stable: minimum-hold and hysteresis rules prevent bouncing, but genuine
   rescue situations override them.
4. Self-attuning: an optional credit system learns which acquisition function
   has historically delivered improvement on *this* dataset; ``robust=True``
   protects all thresholds from outlier yields; ``suggest_config`` calibrates
   a config from your data.
5. Budget-aware: pass ``batches_remaining`` and the switcher explores early
   and exploits in the endgame.
6. Statistically grounded (v5): noise is estimated with the outlier-proof
   MAD, trends with the Theil-Sen slope, hit rates are gated on Wilson
   lower confidence bounds (so tiny batches can't trigger premature
   exploitation), and Sobol restarts require record-value-theory evidence
   that new bests are arriving no faster than random sampling would
   produce. All additions are deterministic and add no new dependencies.
7. Anytime-valid (v5): stagnation and worsening verdicts are confirmed by
   a test-by-betting e-detector (game-theoretic statistics; Ville's
   inequality gives false-alarm control that survives checking after
   every batch, which fixed-threshold rules silently lack). The wealth
   values are exposed in diagnostics as interpretable evidence ratios.
8. Self-contained diversity signal (v5): when no external ``diversity``
   input is supplied, a yield-dispersion proxy (latest batch IQR relative
   to the recent campaign window) detects collapse onto a narrow region,
   and a Wasserstein-1 drift diagnostic tracks distribution shift.

The only function your BO loop needs to call is ``choose_next_acquisition``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

VALID_ACQUISITIONS = {"Sobol", "UCB", "EI", "PI"}
BO_ACQUISITIONS = {"UCB", "EI", "PI"}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SwitchConfig:
    # ---- Basic switching behavior -----------------------------------------
    min_completed_batches_before_switching: int = 3
    min_batches_per_acquisition: int = 2
    patience: int = 2
    rolling_window: int = 3

    # ---- Scale handling -----------------------------------------------------
    # auto_scale=True (recommended): yield thresholds are fractions of the
    # observed yield range, so any objective scale works.
    # auto_scale=False: absolute values below are used verbatim and assume a
    # percent-yield (0-100) scale.
    auto_scale: bool = True
    rel_min_change: float = 0.02   # smallest change considered "real"
    rel_avg_drop: float = 0.05     # per-batch average drop counting as worsening

    # Absolute thresholds, used when auto_scale=False.
    min_real_yield_change: float = 2.0
    avg_drop_tol: float = 5.0
    absolute_hit_yield: Optional[float] = 70.0

    # Noise handling: tolerance is at least noise_multiplier * (spread of
    # recent batch-average deltas).
    noise_multiplier: float = 0.50

    # ---- Statistical estimators (v5) ----------------------------------------
    # use_mad_noise=True: estimate batch-to-batch noise with the MAD
    # (median absolute deviation, scaled by 1.4826 to be std-consistent for
    # Gaussian data) instead of the sample std. One outlier batch then
    # cannot inflate the adaptive tolerance and mask real stagnation.
    use_mad_noise: bool = True

    # Hit rates are judged by their Wilson lower confidence bound rather
    # than the raw point estimate, so 2/4 hits in a tiny batch (raw rate
    # 0.5, but consistent with a true rate of ~15%) cannot trigger
    # premature exploitation. z = 1.28 is a one-sided ~90% bound; set to
    # 0.0 to recover the raw-rate behavior.
    hit_rate_confidence_z: float = 1.28

    # Record-rate gate for Sobol restarts: under purely random sampling the
    # probability that trial n sets a new best is exactly 1/n, so the
    # expected number of new records over recent trials is known. A restart
    # is only justified when records are arriving at or below this random
    # baseline (the surrogate is demonstrably adding nothing). Requires at
    # least record_gate_min_trials recent trials to be conclusive.
    record_gate_min_trials: int = 5

    # ---- Anytime-valid evidence confirmation (e-detector) -------------------------------
    # The patience-window rules are checked after EVERY batch, which makes
    # their effective false-alarm rate grow with campaign length (the
    # classic "peeking" problem). When use_evidence_test=True, stagnation
    # and worsening must additionally be confirmed by a test-by-betting
    # wealth process: each batch multiplies the wealth by a Bernoulli
    # likelihood ratio, and by Ville's inequality the probability the
    # wealth EVER exceeds 1/alpha under the null is at most alpha, no
    # matter how often it is checked. evidence_min_wealth is the required
    # evidence ratio (2.0 = "twice as likely under the stalled hypothesis
    # as under the productive one"); raise it for stronger guarantees at
    # the cost of slower detection. The restart floor (Shin & Ramdas
    # e-detector) keeps a productive early campaign from banking immunity
    # against a later genuine plateau.
    use_evidence_test: bool = True
    evidence_min_wealth: float = 2.0

    # ---- Yield-dispersion diversity proxy ------------------------------------------------
    # When the optional ``diversity`` input is not supplied, derive a
    # stand-in from the data: the latest batch's IQR relative to the IQR of
    # the recent campaign window. A collapse of this ratio means the batch
    # outcomes are concentrated in a narrow band -- evidence the optimizer
    # is hammering one region. Compared against the same low_diversity
    # threshold. A normalized Wasserstein-1 drift between the latest batch
    # and the prior window is also computed (diagnostics only).
    use_dispersion_proxy: bool = True

    # ---- Robust statistics ----------------------------------------------------
    # robust=True: batch medians instead of means, quantile-based yield range
    # (5th-95th) and reference best (98th percentile), so a single outlier
    # yield cannot distort thresholds, stagnation, or hit rates.
    robust: bool = False
    robust_range_quantiles: Tuple[float, float] = (0.05, 0.95)
    robust_best_quantile: float = 0.98
    robust_min_trials: int = 10  # fall back to plain stats below this count

    # ---- Hit-rate definitions ---------------------------------------------------
    hit_fraction_of_best: float = 0.90
    high_hit_rate: float = 0.50
    min_batches_before_PI: int = 3
    high_avg_fraction_of_best: float = 0.80

    # ---- Optional normalized model diagnostics (0-1) ------------------------------
    high_uncertainty: float = 0.60
    low_uncertainty: float = 0.30
    low_diversity: float = 0.25

    # ---- Hysteresis -----------------------------------------------------------------
    switch_score_margin: float = 0.75

    # ---- Initial BO acquisition after Sobol/random -------------------------------------
    first_bo_acquisition: str = "UCB"

    # ---- One acquisition per batch (design contract) ------------------------------------
    # Every batch must be generated by exactly ONE acquisition function;
    # switching happens only between batches, never within one. When True
    # (default), history containing a batch with multiple acquisition labels
    # raises ValueError (it indicates a bug in the BO loop or bookkeeping).
    # Set False to downgrade to a warning and fall back to the most common
    # label in that batch.
    strict_single_acquisition: bool = True

    # ---- Per-acquisition credit (on-the-fly learning) ---------------------------------
    # Each BO acquisition earns a bounded score bonus/penalty based on the
    # RECENCY-WEIGHTED best-yield improvement it has delivered on this
    # dataset (normalized by the adaptive tolerance). credit_decay < 1
    # discounts older batches, so the credit tracks which AF is working NOW
    # and adapts as the campaign moves through phases.
    enable_credit: bool = True
    w_credit: float = 1.0
    credit_min_batches: int = 2   # batches an AF must have run to earn credit
    credit_decay: float = 0.8     # per-batch age discount (1.0 = lifetime mean)

    # ---- Usage balancing (portfolio behavior) -------------------------------------------
    # Soft pressure toward balanced use of UCB/EI/PI over a recent window:
    # under-used AFs earn a bonus, over-used ones a penalty, scaled by
    # w_balance. This encodes the portfolio-BO finding that keeping all
    # three in rotation often outperforms any single AF, while remaining a
    # SOFT prior the data can always override (rescue rules and large score
    # gaps win). Set w_balance=0.0 to disable entirely.
    w_balance: float = 0.5
    balance_window: int = 6

    # ---- Discounted-UCB bandit selection (principled alternative) -----------------------
    # enable_bandit=True replaces the heuristic usage-balance term with the
    # exploration bonus from discounted UCB1 (Garivier & Moulines, 2011):
    #     bonus_i = bandit_c * sqrt(log(N_gamma) / n_gamma_i)
    # where n_gamma_i is the credit_decay-discounted count of batches AF i
    # has run and N_gamma their sum. Unlike the usage-share heuristic, the
    # bonus scales with statistical uncertainty about each AF's value: an
    # AF that ran recently and clearly failed stops being boosted, while an
    # AF that has not run for a while becomes worth re-checking as its
    # discounted count decays. Fully deterministic (argmax, no sampling);
    # pairs with the existing credit term, which supplies the discounted
    # mean-reward half of the UCB index. Rescue rules still take precedence.
    enable_bandit: bool = False
    bandit_c: float = 1.0          # exploration coefficient
    bandit_bonus_cap: float = 2.0  # cap (x bandit_c) for never/rarely-run AFs

    # bandit_prior_strength (tau): how many batches of reward evidence it
    # takes to halve the weight of the heuristic vote prior. The vote
    # scores are multiplied by tau / (tau + n_eff), so at n_eff = tau the
    # prior is at half strength and it fades thereafter. This is the
    # decaying-prior synthesis: the hand-tuned votes give a strong
    # cold-start prior, the discounted-UCB index (credit + bonus) takes
    # over as the campaign delivers evidence. Validated to beat both the
    # static heuristic and the prior-free pure bandit across stationary
    # and non-stationary AF regimes. Larger tau = trust the prior longer.
    bandit_prior_strength: float = 2.0

    # bandit_pure=True is the tau=0 limit: discard the heuristic vote prior
    # entirely and select on the discounted-UCB1 index alone (credit +
    # exploration bonus). Best only when you expect strong non-stationarity
    # from the very first batches; otherwise the default decaying prior is
    # better because it keeps the cold-start prior. Requires enable_bandit.
    bandit_pure: bool = False

    # enable_rotation=True adds SCHEDULED rotation: once the minimum hold for
    # the current AF elapses and no explicit rescue rule has fired, switch to
    # the least-used BO acquisition in the window. This realizes the
    # empirical finding that roughly equal use of UCB/EI/PI can outperform
    # any single AF. Safety guards still apply: rotation never moves INTO PI
    # under high uncertainty, worsening averages, or before
    # min_batches_before_PI; rescue rules always take precedence.
    enable_rotation: bool = False

    # ---- Structural prior on EI (the "balanced default" baseline) -----------------------
    # Set lower (e.g. 0.5) to reduce predisposition toward EI.
    w_ei_prior: float = 1.0

    # ---- Budget awareness (used when batches_remaining is passed) --------------------------
    early_fraction: float = 0.25     # first 25% of budget: exploration bonus
    endgame_fraction: float = 0.25   # last 25% of budget: exploitation bonus
    w_budget: float = 1.5

    # ---- Mistake revert ----------------------------------------------------------------
    # If the FIRST batch after switching to a new acquisition function shows
    # a severe average collapse (worse than revert_drop_multiplier x the
    # adaptive drop tolerance), treat the switch as a mistake and revert
    # immediately to the previous acquisition function, bypassing the
    # minimum-hold and hysteresis rules. The 2x multiplier keeps ordinary
    # noise from triggering false reverts.
    enable_mistake_revert: bool = True
    revert_drop_multiplier: float = 2.0

    # ---- Random-restart escape hatch ------------------------------------------------------
    # If UCB itself stagnates for exploration_patience batches, propose a
    # fresh Sobol batch (model is likely misled everywhere it has data).
    allow_sobol_restart: bool = False
    exploration_patience: int = 3

    # ---- Model-signal phase control (the "switch at the right place" layer) ---------------
    # When the BO loop passes ``model_signal`` (the optimizer's own belief
    # about the unexplored space), the switcher reads two principled BO
    # statistics instead of relying on observed yields alone:
    #
    #   max_ei      -- the maximum Expected Improvement over the untested
    #                  candidate pool, in yield units. This is the standard
    #                  BO convergence signal: when no candidate is expected
    #                  to beat the incumbent by a meaningful margin, the
    #                  exploration phase is statistically finished.
    #   optimism_gap-- (max optimistic upper bound over the pool) minus the
    #                  best observed yield, in yield units; e.g. max over
    #                  candidates of mean + k*sd, minus best-so-far. This is
    #                  how much better the model still BELIEVES is reachable.
    #
    # Together they distinguish the two situations a human cannot tell apart
    # from a yield curve alone:
    #   * CONVERGED  : yield stagnant AND optimism_gap small  -> the model
    #                  agrees nothing better exists; stop exploring, exploit.
    #   * STUCK      : yield stagnant BUT optimism_gap large   -> the model
    #                  thinks better yields exist elsewhere; escape the basin
    #                  with exploration (UCB) or a restart.
    # Thresholds are expressed as multiples of the adaptive yield tolerance,
    # so they are automatically scale- and noise-invariant. The signal is
    # optional; without it the switcher behaves exactly as before.
    use_model_signal: bool = True
    converged_ei_mult: float = 1.0    # max_ei below this*tolerance => converged
    escape_gap_mult: float = 3.0      # optimism_gap above this*tolerance => stuck

    # ---- Prolonged-stall escape (yield-only, no model signal needed) -----------------------
    # Campaign-6 failure mode: best-so-far flat for many consecutive batches
    # while the switcher keeps exploiting/holding. Without model_signal we
    # cannot tell "converged at the optimum" from "stuck in a sub-optimal
    # basin", but after a LONG plateau the right move is the same either way:
    # force a deliberate exploration/restart. If we were converged, one
    # exploratory batch costs little; if we were stuck, it is the only way
    # out. Triggers when the best has not improved for >= stall_escape_patience
    # consecutive batches (longer than ordinary `patience`, so it only fires
    # on a genuine prolonged plateau, not a normal pause).
    enable_stall_escape: bool = True
    stall_escape_patience: int = 4

    # ---- Score weights -------------------------------------------------------------------
    w_uncertainty: float = 2.0
    w_improvement: float = 2.0
    w_stagnation: float = 2.0
    w_hit_rate: float = 2.0
    w_diversity: float = 1.5
    w_avg_drop: float = 2.0

    def __post_init__(self) -> None:
        self.first_bo_acquisition = normalize_acquisition(self.first_bo_acquisition)
        if self.first_bo_acquisition not in BO_ACQUISITIONS:
            raise ValueError(
                f"first_bo_acquisition must be one of {sorted(BO_ACQUISITIONS)}, "
                f"got {self.first_bo_acquisition!r}"
            )
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.rolling_window < self.patience:
            raise ValueError("rolling_window must be >= patience")
        if self.min_batches_per_acquisition < 1:
            raise ValueError("min_batches_per_acquisition must be >= 1")
        if not (0.0 < self.hit_fraction_of_best <= 1.0):
            raise ValueError("hit_fraction_of_best must be in (0, 1]")
        if not (0.0 < self.high_avg_fraction_of_best <= 1.0):
            raise ValueError("high_avg_fraction_of_best must be in (0, 1]")
        q_lo, q_hi = self.robust_range_quantiles
        if not (0.0 <= q_lo < q_hi <= 1.0):
            raise ValueError("robust_range_quantiles must satisfy 0 <= lo < hi <= 1")
        if not (0.0 < self.early_fraction < 1.0 and 0.0 < self.endgame_fraction < 1.0):
            raise ValueError("early_fraction and endgame_fraction must be in (0, 1)")
        if not (0.0 < self.credit_decay <= 1.0):
            raise ValueError("credit_decay must be in (0, 1]")
        if self.w_balance < 0:
            raise ValueError("w_balance must be >= 0")

    @classmethod
    def portfolio(cls, **overrides) -> "SwitchConfig":
        """
        Preset for balanced-rotation behavior: keeps UCB, EI, and PI in
        roughly equal use unless the data strongly argues otherwise.

        Use when (as some campaigns find) no single acquisition function
        dominates and equal rotation performs best. Strong signals
        (stagnation, diversity collapse, rescue rules) still override the
        rotation pressure.
        """
        defaults: Dict[str, Any] = dict(
            w_balance=2.0,
            w_ei_prior=0.5,
            min_batches_before_PI=3,
            switch_score_margin=0.5,
            enable_rotation=True,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def adaptive(cls, **overrides) -> "SwitchConfig":
        """Default signal-driven behavior with mild balance pressure."""
        return cls(**overrides)


# Backwards-compatible alias for existing code.
SwitchConfigV2 = SwitchConfig


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def normalize_acquisition(acq: Any) -> str:
    """Normalize common acquisition-function names into Sobol, UCB, EI, or PI."""
    if acq is None or (isinstance(acq, float) and np.isnan(acq)):
        return "Unknown"

    s = str(acq).strip().lower().replace("-", "_").replace(" ", "_")

    if "sobol" in s or "random" in s or s in {"lhs", "latin_hypercube"}:
        return "Sobol"
    if "ucb" in s or "upper_confidence" in s or "upperconfidence" in s:
        return "UCB"
    if s == "ei" or "expected_improvement" in s or "expectedimprovement" in s \
            or s in {"qei", "q_ei", "logei", "log_ei", "qlogei"}:
        return "EI"
    if s == "pi" or "probability_of_improvement" in s or "probabilityofimprovement" in s \
            or "probability_improvement" in s or s in {"qpi", "q_pi"}:
        return "PI"

    return str(acq)


def _batch_mode(series: pd.Series) -> str:
    """Most common acquisition label within a batch (robust to mixed labels)."""
    modes = series.mode()
    return modes.iloc[0] if not modes.empty else "Unknown"


def summarize_batches(
    history: pd.DataFrame,
    batch_col: str = "batch",
    yield_col: str = "yield",
    acq_col: str = "acquisition",
) -> pd.DataFrame:
    """
    Turn trial-level data into batch-level diagnostics.

    Required columns: ``batch_col``, ``yield_col``. Optional: ``acq_col``.
    """
    if batch_col not in history.columns:
        raise ValueError(f"Missing batch column: {batch_col!r}")
    if yield_col not in history.columns:
        raise ValueError(f"Missing yield column: {yield_col!r}")

    df = history.copy()
    df[yield_col] = pd.to_numeric(df[yield_col], errors="coerce")
    df = df.dropna(subset=[yield_col])

    if df.empty:
        raise ValueError("No valid numeric yield values found.")

    if acq_col not in df.columns:
        df[acq_col] = "Unknown"
    df[acq_col] = df[acq_col].apply(normalize_acquisition)

    batch_summary = (
        df.groupby(batch_col, sort=True)
        .agg(
            batch_average_yield=(yield_col, "mean"),
            batch_best_yield=(yield_col, "max"),
            batch_min_yield=(yield_col, "min"),
            batch_median_yield=(yield_col, "median"),
            batch_std_yield=(yield_col, "std"),
            n_experiments=(yield_col, "count"),
            acquisition_used=(acq_col, _batch_mode),
            n_acquisition_labels=(
                acq_col,
                lambda x: int(x[x != "Unknown"].nunique()),
            ),
        )
        .reset_index()
        .sort_values(batch_col)
        .reset_index(drop=True)
    )

    batch_summary["batch_std_yield"] = batch_summary["batch_std_yield"].fillna(0.0)
    batch_summary["best_yield_so_far"] = batch_summary["batch_best_yield"].cummax()
    batch_summary["worst_yield_so_far"] = batch_summary["batch_min_yield"].cummin()
    batch_summary["delta_best"] = batch_summary["best_yield_so_far"].diff().fillna(0.0)
    batch_summary["delta_batch_average"] = (
        batch_summary["batch_average_yield"].diff().fillna(0.0)
    )
    batch_summary["delta_batch_median"] = (
        batch_summary["batch_median_yield"].diff().fillna(0.0)
    )

    return batch_summary


def _rolling_slope(values: np.ndarray) -> float:
    """
    Theil-Sen slope over ordered recent values (positive = improving).

    The median of all pairwise slopes (y_j - y_i) / (j - i). Robust: one
    outlier batch in the window cannot flip the trend sign, unlike the
    least-squares slope it replaces. Identical cost at these window sizes.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return 0.0

    slopes = [
        (values[j] - values[i]) / float(j - i)
        for i in range(n - 1)
        for j in range(i + 1, n)
        if np.isfinite(values[i]) and np.isfinite(values[j])
    ]
    if not slopes:
        return 0.0
    return float(np.median(slopes))


def _robust_noise(values: np.ndarray) -> float:
    """
    Scale-consistent MAD noise estimate: 1.4826 * median(|x - median(x)|).

    Equals the std for Gaussian data but has a 50% breakdown point, so a
    single wild batch delta cannot inflate it. Falls back to the sample std
    when the MAD degenerates to zero (e.g. heavily tied deltas).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    mad = float(np.median(np.abs(values - np.median(values))))
    if mad > 0:
        return 1.4826 * mad
    std = float(np.std(values, ddof=1))
    return std if np.isfinite(std) else 0.0


def _wilson_lower_bound(p_hat: float, n: int, z: float) -> float:
    """
    Wilson score lower confidence bound for a binomial proportion.

    With tiny batches the raw hit rate is nearly meaningless (2/4 hits is
    consistent with a true rate anywhere from ~15% to ~85%); gating on the
    lower bound makes 'high hit rate' mean 'high hit rate with evidence'.
    z=0 returns the raw rate.
    """
    if n <= 0:
        return 0.0
    if z <= 0:
        return float(p_hat)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p_hat + z2 / (2.0 * n)
    margin = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    return float(max((center - margin) / denom, 0.0))


def _record_rate_deficit(
    history: pd.DataFrame,
    batch_summary: pd.DataFrame,
    batch_col: str,
    yield_col: str,
    window_batches: int,
    min_trials: int,
) -> Tuple[Optional[bool], Dict[str, float]]:
    """
    Test recent new-best records against the random-sampling null.

    Under i.i.d. random sampling, P(trial n sets a new record) = 1/n, so the
    expected number of records among recent trials n0+1..N is
    sum(1/n for n in n0+1..N) regardless of the yield distribution
    (a distribution-free result from record-value theory). Returns
    (deficit, info): deficit=True means observed records <= the random
    baseline, i.e. the surrogate is currently adding nothing over random
    search; None means not enough recent trials to judge.
    """
    df = history[[batch_col, yield_col]].copy()
    df[yield_col] = pd.to_numeric(df[yield_col], errors="coerce")
    df = df.dropna(subset=[yield_col]).sort_values(batch_col, kind="stable")
    y = df[yield_col].to_numpy(dtype=float)
    total = len(y)

    recent_ids = set(batch_summary[batch_col].tail(window_batches).tolist())
    in_window = df[batch_col].isin(recent_ids).to_numpy()
    n_recent = int(in_window.sum())
    n0 = total - n_recent

    if n_recent < min_trials or n0 < 1:
        return None, {}

    prior_max = float(np.max(y[~in_window]))
    running = prior_max
    observed = 0
    for v in y[in_window]:
        if v > running:
            observed += 1
            running = v

    expected = float(sum(1.0 / n for n in range(n0 + 1, total + 1)))
    info = {
        "recent_records_observed": float(observed),
        "recent_records_expected_random": expected,
    }
    return observed <= expected, info


def _betting_evidence(
    indicators: np.ndarray,
    p_null: float,
    p_alt: float,
) -> float:
    """
    E-detector wealth (test-by-betting with restarts) for one-sided drift.

    Each batch multiplies a nonnegative wealth by the likelihood ratio
        e_t = (p_alt/p_null)^b * ((1-p_alt)/(1-p_null))^(1-b)
    for binary indicator b. With p_alt > p_null, the wealth is a
    supermartingale whenever the true indicator rate is <= p_null, so by
    Ville's inequality P(wealth ever >= 1/alpha) <= alpha -- a false-alarm
    guarantee that holds even though the rule is checked after every
    batch, which fixed-threshold rules silently lack. Flooring at 1 before
    each update (the Shin-Ramdas e-detector restart) stops long calm
    stretches from banking evidence against a later genuine change.
    Deterministic; the returned wealth is an interpretable evidence ratio.
    """
    if not (0.0 < p_null < 1.0 and 0.0 < p_alt < 1.0) or p_alt <= p_null:
        return 1.0
    up = p_alt / p_null
    down = (1.0 - p_alt) / (1.0 - p_null)
    wealth = 1.0
    for b in np.asarray(indicators, dtype=bool):
        wealth = max(wealth, 1.0) * (up if b else down)
        wealth = min(wealth, 1e9)
    return float(wealth)


def _dispersion_and_drift(
    history: pd.DataFrame,
    batch_summary: pd.DataFrame,
    batch_col: str,
    yield_col: str,
    yrange: float,
    window: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (dispersion_ratio, w1_drift) for the latest batch, or Nones.

    dispersion_ratio = IQR(latest batch yields) / IQR(yields from the
    previous ``window`` batches). A small ratio means the latest batch's
    outcomes are concentrated in a narrow band relative to what the
    campaign was recently producing -- a data-driven proxy for candidate
    diversity collapse. Comparing against the *recent* window (not the
    whole campaign) avoids false alarms from the natural late-campaign
    concentration around good regions.

    w1_drift = Wasserstein-1 distance between the latest batch's yield
    distribution and the prior window's (computed on a quantile grid),
    normalized by the yield range. Detects whole-distribution shift that
    a difference of means misses (diagnostics only).
    """
    ids = batch_summary[batch_col].tolist()
    if len(ids) < 2 or yrange <= 0:
        return None, None

    df = history[[batch_col, yield_col]].copy()
    df[yield_col] = pd.to_numeric(df[yield_col], errors="coerce")
    df = df.dropna(subset=[yield_col])

    y_last = df.loc[df[batch_col] == ids[-1], yield_col].to_numpy(dtype=float)
    prior_ids = ids[max(len(ids) - 1 - window, 0):-1]
    y_prior = df.loc[df[batch_col].isin(prior_ids), yield_col].to_numpy(dtype=float)

    if len(y_last) < 3 or len(y_prior) < 3:
        return None, None

    def iqr(v: np.ndarray) -> float:
        return float(np.quantile(v, 0.75) - np.quantile(v, 0.25))

    prior_iqr = iqr(y_prior)
    ratio = (iqr(y_last) / prior_iqr) if prior_iqr > 0 else None

    qs = np.linspace(0.05, 0.95, 19)
    w1 = float(
        np.mean(np.abs(np.quantile(y_last, qs) - np.quantile(y_prior, qs)))
    ) / yrange
    return ratio, w1


def _count_batches_since_switch(batch_summary: pd.DataFrame, current: str) -> int:
    """Consecutive completed batches (from the end) that used ``current``."""
    count = 0
    for acq in reversed(batch_summary["acquisition_used"].tolist()):
        if acq == current:
            count += 1
        else:
            break
    return count


def _previous_acquisition(batch_summary: pd.DataFrame, current: str) -> Optional[str]:
    """The acquisition used immediately before the current streak, if any."""
    for acq in reversed(batch_summary["acquisition_used"].tolist()):
        if acq != current:
            return acq
    return None


def _reference_scale(
    history: pd.DataFrame,
    batch_summary: pd.DataFrame,
    yield_col: str,
    config: SwitchConfig,
) -> Tuple[float, float, bool]:
    """
    Return (reference_best, yield_range, robust_active).

    Plain mode: reference_best = observed max, range = max - min.
    Robust mode (enough trials): reference_best = 98th-percentile yield and
    range = 95th - 5th percentile, so one outlier cannot distort thresholds.
    """
    best = float(batch_summary["best_yield_so_far"].iloc[-1])
    worst = float(batch_summary["worst_yield_so_far"].iloc[-1])

    yvals = pd.to_numeric(history[yield_col], errors="coerce").dropna().to_numpy(dtype=float)

    if config.robust and len(yvals) >= config.robust_min_trials:
        q_lo, q_hi = config.robust_range_quantiles
        # Inward-biased quantile methods ('higher' for the floor, 'lower' for
        # the ceiling) land on actual bulk data points instead of
        # interpolating toward an outlier — critical for small datasets.
        lo = float(np.quantile(yvals, q_lo, method="higher"))
        hi = float(np.quantile(yvals, q_hi, method="lower"))
        ref_best = float(np.quantile(yvals, config.robust_best_quantile, method="lower"))
        ref_best = max(ref_best, hi)
        yrange = max(hi - lo, 0.0)
        if yrange > 0:
            return ref_best, yrange, True

    return best, max(best - worst, 0.0), False


def _adaptive_yield_tolerance(
    batch_summary: pd.DataFrame,
    config: SwitchConfig,
    yrange: float,
    delta_avg_col: str,
) -> float:
    """
    Estimate a meaningful yield-improvement threshold: the larger of a base
    threshold (relative to the observed range when auto_scale=True) and a
    noise term from recent batch-to-batch variation.
    """
    if config.auto_scale and yrange > 0:
        base = config.rel_min_change * yrange
    else:
        base = config.min_real_yield_change

    # Drop the artificial first delta (diff().fillna(0)) so it does not bias
    # the noise estimate toward zero.
    deltas = batch_summary[delta_avg_col].iloc[1:]
    recent_delta = deltas.tail(max(config.rolling_window + 1, 3))

    observed_noise = 0.0
    if len(recent_delta) >= 2:
        arr = recent_delta.to_numpy(dtype=float)
        if config.use_mad_noise:
            observed_noise = _robust_noise(arr)
        else:
            observed_noise = float(np.nanstd(arr, ddof=1))
        if not np.isfinite(observed_noise):
            observed_noise = 0.0

    tolerance = max(base, config.noise_multiplier * observed_noise)
    if not np.isfinite(tolerance) or tolerance <= 0:
        tolerance = base if base > 0 else 1e-9
    return float(tolerance)


def _avg_drop_tolerance(config: SwitchConfig, yrange: float) -> float:
    """How large a per-batch average decline counts as 'worsening'."""
    if config.auto_scale and yrange > 0:
        return config.rel_avg_drop * yrange
    return config.avg_drop_tol


def _hit_threshold(ref_best: float, yrange: float, config: SwitchConfig) -> float:
    """
    Yield value above which a trial counts as a "hit".

    auto_scale=True: ref_best - (1 - hit_fraction_of_best) * yield_range.
    Offset- and scale-invariant (works for 0-1, 0-100, negative objectives).

    auto_scale=False: legacy percent-scale rule
    max(absolute_hit_yield, hit_fraction_of_best * ref_best).
    """
    if config.auto_scale:
        if yrange > 0:
            return ref_best - (1.0 - config.hit_fraction_of_best) * yrange
        return ref_best  # all yields identical so far

    threshold = config.hit_fraction_of_best * ref_best
    if config.absolute_hit_yield is not None:
        threshold = max(config.absolute_hit_yield, threshold)
    return threshold


def _hit_rate_for_batches(
    history: pd.DataFrame,
    batch_ids: list,
    batch_col: str,
    yield_col: str,
    threshold: float,
) -> Tuple[float, int]:
    """Return (raw hit rate, number of trials) for the given batches."""
    subset = history[history[batch_col].isin(batch_ids)].copy()
    subset[yield_col] = pd.to_numeric(subset[yield_col], errors="coerce")
    subset = subset.dropna(subset=[yield_col])
    if subset.empty:
        return 0.0, 0
    return float((subset[yield_col] >= threshold).mean()), int(len(subset))


def _high_batch_average(
    batch_avg: float,
    ref_best: float,
    yrange: float,
    config: SwitchConfig,
) -> bool:
    """Is the latest batch average 'close to the best'? Offset-invariant."""
    if config.auto_scale:
        if yrange <= 0:
            return True
        floor = ref_best - yrange
        return (batch_avg - floor) >= config.high_avg_fraction_of_best * yrange
    return batch_avg >= config.high_avg_fraction_of_best * max(ref_best, 1e-9)


def _acquisition_credit(
    batch_summary: pd.DataFrame,
    tolerance: float,
    config: SwitchConfig,
) -> Dict[str, float]:
    """
    Recency-weighted credit: the best-yield improvement each BO acquisition
    has recently delivered on this dataset.

    Each batch's reward is its best-yield gain normalized by the adaptive
    tolerance (clipped to [-1, 1]), then rewards are discounted by
    ``credit_decay`` per batch of age. The credit therefore tracks which AF
    is working NOW — an AF that was productive early but has gone cold loses
    its advantage — which is what lets the switcher relearn on the fly as a
    campaign moves between smooth and rough regions of chemistry space.
    """
    credits: Dict[str, float] = {}
    df = batch_summary.iloc[1:]  # first batch's delta is an artifact
    n = len(df)
    if n == 0:
        return credits

    ages = np.arange(n - 1, -1, -1, dtype=float)  # most recent batch: age 0
    weights_all = np.power(config.credit_decay, ages)
    rewards_all = np.clip(
        df["delta_best"].to_numpy(dtype=float) / max(tolerance, 1e-12), -1.0, 1.0
    )
    labels = df["acquisition_used"].to_numpy()

    for af in BO_ACQUISITIONS:
        mask = labels == af
        if int(mask.sum()) >= config.credit_min_batches:
            w = weights_all[mask]
            credits[af] = float(np.sum(w * rewards_all[mask]) / max(float(np.sum(w)), 1e-12))
    return credits


def _bandit_exploration_bonus(
    batch_summary: pd.DataFrame,
    config: SwitchConfig,
) -> Dict[str, float]:
    """
    Discounted-UCB1 exploration bonus per acquisition function.

    bonus_i = bandit_c * sqrt(log(N_gamma) / n_gamma_i), with
    n_gamma_i = sum of credit_decay^age over batches AF i generated and
    N_gamma = sum over all BO AFs. The bonus is large for AFs with little
    recent evidence and shrinks as evidence accumulates; discounting makes
    stale evidence expire, so a long-unused AF gradually becomes worth
    re-checking. Together with the discounted-mean credit term this forms
    the full discounted-UCB index of Garivier & Moulines (2011).
    Deterministic; capped at bandit_c * bandit_bonus_cap for AFs with
    (near-)zero discounted count.
    """
    df = batch_summary.iloc[1:]
    labels = df["acquisition_used"].to_numpy()
    n = len(df)
    cap = config.bandit_c * config.bandit_bonus_cap

    if n == 0:
        return {af: cap for af in BO_ACQUISITIONS}

    ages = np.arange(n - 1, -1, -1, dtype=float)
    weights = np.power(config.credit_decay, ages)

    counts = {
        af: float(np.sum(weights[labels == af])) for af in BO_ACQUISITIONS
    }
    total = max(sum(counts.values()), 1e-12)
    log_total = math.log(max(total, math.e))  # >= 1 so the bonus never vanishes

    bonuses: Dict[str, float] = {}
    for af, n_gamma in counts.items():
        if n_gamma <= 1e-9:
            bonuses[af] = cap
        else:
            bonuses[af] = float(
                min(config.bandit_c * math.sqrt(log_total / n_gamma), cap)
            )
    return bonuses


def _usage_balance(
    batch_summary: pd.DataFrame,
    config: SwitchConfig,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Soft pressure toward balanced use of UCB/EI/PI over a recent window.

    Returns (bonuses, usage_shares). Under-used AFs get a positive bonus
    proportional to their usage deficit; over-used AFs a penalty; both
    bounded by w_balance. Sobol batches are excluded from the shares.
    """
    target = 1.0 / 3.0
    recent = batch_summary["acquisition_used"].tail(config.balance_window)
    bo = recent[recent.isin(BO_ACQUISITIONS)]

    shares: Dict[str, float] = {}
    bonuses: Dict[str, float] = {}
    for af in BO_ACQUISITIONS:
        share = float((bo == af).mean()) if len(bo) else target
        shares[af] = share
        raw = config.w_balance * (target - share) / target
        bonuses[af] = float(np.clip(raw, -config.w_balance, config.w_balance))
    return bonuses, shares


def _data_warnings(
    batch_summary: pd.DataFrame,
    yrange: float,
    mean_uncertainty: Optional[float],
    diversity: Optional[float],
) -> List[str]:
    """Lightweight data-quality checks surfaced with every decision."""
    warnings: List[str] = []
    sizes = batch_summary["n_experiments"]

    if float(sizes.median()) < 3:
        warnings.append(
            "Median batch size < 3: batch averages and hit rates are very "
            "noisy; consider larger patience or robust=True."
        )
    if len(sizes) > 1 and float(sizes.mean()) > 0:
        cv = float(sizes.std(ddof=0)) / float(sizes.mean())
        if cv > 0.5:
            warnings.append(
                "Batch sizes vary widely; batch-average trends may reflect "
                "batch size rather than chemistry."
            )
    if yrange <= 0:
        warnings.append(
            "All observed yields are identical; thresholds are degenerate "
            "and decisions default to exploration-leaning behavior."
        )
    if mean_uncertainty is not None and not (0.0 <= float(mean_uncertainty) <= 1.0):
        warnings.append(
            f"mean_uncertainty={mean_uncertainty} is outside [0, 1]; "
            "normalize it (e.g. mean posterior std / observed yield range)."
        )
    if diversity is not None and not (0.0 <= float(diversity) <= 1.0):
        warnings.append(f"diversity={diversity} is outside [0, 1]; normalize it.")
    return warnings


# --------------------------------------------------------------------------- #
# Main decision function
# --------------------------------------------------------------------------- #
def choose_next_acquisition(
    history: pd.DataFrame,
    current_acquisition: Optional[str] = None,
    batch_col: str = "batch",
    yield_col: str = "yield",
    acq_col: str = "acquisition",
    mean_uncertainty: Optional[float] = None,
    diversity: Optional[float] = None,
    config: Optional[SwitchConfig] = None,
    batches_remaining: Optional[int] = None,
    model_signal: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Decide which acquisition function should be used for the next full batch.

    Design contract: every batch is generated by exactly ONE acquisition
    function, and the decision returned here applies to the entire next
    batch. Switching only ever happens between batches (after each run
    completes), never within a batch. History violating this raises
    ValueError unless ``strict_single_acquisition=False``.

    This is the only function your BO code needs to call after each batch
    completes (it consumes data incrementally as the campaign progresses).

    Parameters
    ----------
    history:
        All completed experiments so far. One row per trial. Must contain
        ``batch_col`` and ``yield_col``; ``acq_col`` is optional.
    current_acquisition:
        Acquisition used most recently. If None, inferred from the latest batch.
    mean_uncertainty:
        Optional normalized model uncertainty in [0, 1].
    diversity:
        Optional normalized candidate diversity in [0, 1].
    config:
        SwitchConfig instance. A fresh default is created when None.
    batches_remaining:
        Optional number of batches left in the campaign budget. When given,
        scores ramp toward exploration early and exploitation late, and
        endgame rescues prefer EI over UCB.
    model_signal:
        Optional dict carrying the surrogate's own belief about the
        unexplored space, computed by the BO loop from the same posterior
        it uses to score candidates. Recognized keys (both in yield units):
        ``max_ei`` (maximum Expected Improvement over untested candidates)
        and ``optimism_gap`` (best optimistic upper bound over the pool minus
        the best observed yield). These drive the converged-vs-stuck phase
        logic; omit to keep pure yield-driven behavior.

    Returns
    -------
    dict with keys: next_acquisition, reason, state, diagnostics, scores,
    warnings, batch_summary.
    """
    if config is None:
        config = SwitchConfig()

    batch_summary = summarize_batches(history, batch_col, yield_col, acq_col)
    n_batches = len(batch_summary)
    last = batch_summary.iloc[-1]

    # ---- Contract: exactly one acquisition function per batch -------------------
    # Switching happens only BETWEEN batches. A batch carrying multiple
    # acquisition labels means the BO loop split a batch across AFs or the
    # bookkeeping is wrong; every batch-level statistic would be unreliable.
    extra_warnings: List[str] = []
    mixed = batch_summary[batch_summary["n_acquisition_labels"] > 1]
    if not mixed.empty:
        mixed_ids = mixed[batch_col].tolist()
        msg = (
            f"Batches {mixed_ids} contain multiple acquisition labels, but each "
            f"batch must be generated by exactly one acquisition function "
            f"(switching only happens between batches). Fix the BO loop or "
            f"trial labels, or set strict_single_acquisition=False to fall "
            f"back to the most common label per batch."
        )
        if config.strict_single_acquisition:
            raise ValueError(msg)
        extra_warnings.append(msg)

    if current_acquisition is None:
        current_acquisition = last["acquisition_used"]
    current_acquisition = normalize_acquisition(current_acquisition)

    if current_acquisition not in VALID_ACQUISITIONS:
        valid = [
            a for a in batch_summary["acquisition_used"].tolist()
            if a in VALID_ACQUISITIONS
        ]
        current_acquisition = valid[-1] if valid else "EI"

    # ---- Initial phase ---------------------------------------------------------
    if n_batches < config.min_completed_batches_before_switching:
        next_acq = (
            current_acquisition
            if current_acquisition in BO_ACQUISITIONS
            else config.first_bo_acquisition
        )
        return {
            "next_acquisition": next_acq,
            "reason": (
                f"Too few completed batches for adaptive switching "
                f"({n_batches}/{config.min_completed_batches_before_switching}); "
                f"use {next_acq}."
            ),
            "state": "initial_bo",
            "diagnostics": {
                "n_completed_batches": n_batches,
                "current_acquisition": current_acquisition,
            },
            "scores": {"UCB": np.nan, "EI": np.nan, "PI": np.nan},
            "warnings": extra_warnings,
            "batch_summary": batch_summary,
        }

    # ---- Statistic selection (robust mode uses medians) ---------------------------
    avg_col = "batch_median_yield" if config.robust else "batch_average_yield"
    delta_avg_col = "delta_batch_median" if config.robust else "delta_batch_average"

    ref_best, yrange, robust_active = _reference_scale(
        history, batch_summary, yield_col, config
    )

    # ---- Diagnostics ---------------------------------------------------------------
    tolerance = _adaptive_yield_tolerance(batch_summary, config, yrange, delta_avg_col)
    avg_drop_tol = _avg_drop_tolerance(config, yrange)

    recent = batch_summary.tail(config.rolling_window)
    recent_best_deltas = recent["delta_best"].to_numpy(dtype=float)
    recent_avg_deltas = recent[delta_avg_col].to_numpy(dtype=float)

    best_yield = float(last["best_yield_so_far"])
    batch_avg = float(last[avg_col])
    delta_best = float(last["delta_best"])
    delta_avg = float(last[delta_avg_col])

    avg_slope = _rolling_slope(recent[avg_col].to_numpy(dtype=float))
    best_slope = _rolling_slope(recent["best_yield_so_far"].to_numpy(dtype=float))

    stagnated_run_rule = bool(
        len(recent_best_deltas) >= config.patience
        and np.all(recent_best_deltas[-config.patience:] < tolerance)
    )
    avg_worsening_run_rule = bool(
        len(recent_avg_deltas) >= config.patience
        and np.all(recent_avg_deltas[-config.patience:] <= -avg_drop_tol)
    )

    # Prolonged stall: best-so-far has not improved for a LONG run of batches
    # (longer than ordinary patience). Yield-only signal used by the
    # model-free escape rule for the campaign-6 stuck-forever failure mode.
    all_best_deltas_full = batch_summary["delta_best"].iloc[1:].to_numpy(dtype=float)
    prolonged_stall = bool(
        len(all_best_deltas_full) >= config.stall_escape_patience
        and np.all(all_best_deltas_full[-config.stall_escape_patience:] < tolerance)
    )

    # ---- Anytime-valid confirmation (e-detector) -------------------------------
    # The run rules above are evaluated after every batch, so their false-
    # alarm rate grows with campaign length. Confirm them with betting
    # wealth: stagnation bets on "no real improvement" batches against the
    # campaign's own pre-window improvement rate; worsening bets on
    # significant-decline batches against a low base rate.
    all_best_deltas = batch_summary["delta_best"].iloc[1:].to_numpy(dtype=float)
    all_avg_deltas = batch_summary[delta_avg_col].iloc[1:].to_numpy(dtype=float)

    quiet = all_best_deltas < tolerance
    pre_window = quiet[:-config.patience] if len(quiet) > config.patience else quiet[:0]
    if len(pre_window) > 0:
        improve_rate = 1.0 - float(np.mean(pre_window))
    else:
        improve_rate = 0.5
    p_quiet_null = 1.0 - float(np.clip(improve_rate, 1.0 / 3.0, 0.75))
    stagnation_wealth = _betting_evidence(
        quiet, p_quiet_null, p_quiet_null + (1.0 - p_quiet_null) / 2.0
    )
    worsening_wealth = _betting_evidence(
        all_avg_deltas <= -avg_drop_tol, 0.2, 0.5
    )

    if config.use_evidence_test:
        stagnated = stagnated_run_rule and stagnation_wealth >= config.evidence_min_wealth
        avg_worsening = (
            avg_worsening_run_rule and worsening_wealth >= config.evidence_min_wealth
        )
    else:
        stagnated = stagnated_run_rule
        avg_worsening = avg_worsening_run_rule

    hit_threshold = _hit_threshold(ref_best, yrange, config)
    latest_hit_rate, latest_hit_n = _hit_rate_for_batches(
        history, [last[batch_col]], batch_col, yield_col, hit_threshold
    )
    recent_hit_rate, recent_hit_n = _hit_rate_for_batches(
        history,
        batch_summary[batch_col].tail(config.rolling_window).tolist(),
        batch_col,
        yield_col,
        hit_threshold,
    )

    # Wilson lower bounds: 'high hit rate' must be supported by enough
    # trials, not just a lucky 2-of-4 batch, before PI earns credit.
    latest_hit_lcb = _wilson_lower_bound(
        latest_hit_rate, latest_hit_n, config.hit_rate_confidence_z
    )
    recent_hit_lcb = _wilson_lower_bound(
        recent_hit_rate, recent_hit_n, config.hit_rate_confidence_z
    )

    high_hit_rate = latest_hit_lcb >= config.high_hit_rate
    recent_high_hit_rate = recent_hit_lcb >= config.high_hit_rate
    high_batch_average = _high_batch_average(batch_avg, ref_best, yrange, config)

    high_uncertainty = (
        mean_uncertainty is not None
        and float(mean_uncertainty) >= config.high_uncertainty
    )
    low_uncertainty = (
        mean_uncertainty is not None
        and float(mean_uncertainty) <= config.low_uncertainty
    )
    unknown_uncertainty = mean_uncertainty is None

    # ---- Diversity: external input, or yield-dispersion proxy -------------------
    dispersion_ratio, w1_drift = _dispersion_and_drift(
        history, batch_summary, batch_col, yield_col, yrange, config.rolling_window
    )
    diversity_proxy_active = False
    if diversity is not None:
        low_diversity = float(diversity) <= config.low_diversity
    elif config.use_dispersion_proxy and dispersion_ratio is not None:
        low_diversity = dispersion_ratio <= config.low_diversity
        diversity_proxy_active = True
    else:
        low_diversity = False
    unknown_diversity = diversity is None and not diversity_proxy_active

    batches_since_switch = _count_batches_since_switch(
        batch_summary, current_acquisition
    )

    # Record-value test: are new bests still arriving faster than random
    # sampling would produce? Used to gate Sobol restarts and as a
    # diagnostic for whether the surrogate is adding value at all.
    record_deficit, record_info = _record_rate_deficit(
        history,
        batch_summary,
        batch_col,
        yield_col,
        config.rolling_window,
        config.record_gate_min_trials,
    )

    slope_tol = tolerance / max(config.rolling_window - 1, 1)
    improving_best = delta_best >= tolerance or best_slope >= slope_tol
    improving_avg = delta_avg >= tolerance or avg_slope >= slope_tol
    improving = improving_best or improving_avg

    # ---- Budget progress -------------------------------------------------------------
    progress: Optional[float] = None
    endgame = False
    early_phase = False
    if batches_remaining is not None and batches_remaining >= 0:
        total = n_batches + batches_remaining
        if total > 0:
            progress = n_batches / total
            endgame = progress >= 1.0 - config.endgame_fraction
            early_phase = progress <= config.early_fraction

    # ---- Model-signal phase statistics (converged vs stuck) -----------------------------
    # Read the optimizer's own belief about the unexplored space. These are
    # the principled "is now the right place to switch" statistics: the
    # explore->exploit transition is driven by the model expecting no
    # further improvement, and the escape trigger fires only when the model
    # still believes better yields are reachable despite a yield plateau.
    model_converged = False     # model expects nothing left worth chasing
    stuck_in_local_opt = False  # yield plateau, but model is still optimistic
    max_ei = optimism_gap = None
    if config.use_model_signal and model_signal:
        max_ei = model_signal.get("max_ei")
        optimism_gap = model_signal.get("optimism_gap")
        tol = max(tolerance, 1e-12)
        if max_ei is not None and np.isfinite(max_ei):
            model_converged = float(max_ei) < config.converged_ei_mult * tol
        if optimism_gap is not None and np.isfinite(optimism_gap):
            stuck_in_local_opt = (
                stagnated and float(optimism_gap) > config.escape_gap_mult * tol
            )

    # ---- Scores -------------------------------------------------------------------------
    scores = {"UCB": 0.0, "EI": 0.0, "PI": 0.0}

    # UCB: exploration and rescue
    if high_uncertainty:
        scores["UCB"] += config.w_uncertainty
    if stagnated:
        scores["UCB"] += config.w_stagnation
    if avg_worsening:
        scores["UCB"] += config.w_avg_drop
    if low_diversity:
        scores["UCB"] += config.w_diversity
    if unknown_uncertainty and stagnated:
        scores["UCB"] += 0.5

    # EI: balanced default
    scores["EI"] += config.w_ei_prior
    if improving:
        scores["EI"] += config.w_improvement
    if unknown_uncertainty:
        scores["EI"] += 0.5
    if not high_uncertainty and not low_uncertainty:
        scores["EI"] += 0.5
    if stagnated and not avg_worsening:
        scores["EI"] += 0.5

    # PI: exploitation
    pi_banned = n_batches < config.min_batches_before_PI
    if not pi_banned:
        if low_uncertainty:
            scores["PI"] += config.w_uncertainty
        if high_hit_rate:
            scores["PI"] += config.w_hit_rate
        if recent_high_hit_rate:
            scores["PI"] += config.w_hit_rate
        if high_batch_average:
            scores["PI"] += 1.0
        if improving_best and high_batch_average:
            scores["PI"] += 0.5

    # During rapid improvement the latest batch always sits near the new
    # best-so-far, which inflates hit rates and can trigger premature
    # exploitation. Unless uncertainty is confirmed low, keep searching
    # while the campaign is still climbing.
    if improving and not low_uncertainty:
        scores["PI"] -= 1.0

    # Penalize PI when warning signals appear
    if high_uncertainty:
        scores["PI"] -= 2.0
    if stagnated:
        scores["PI"] -= 2.0
    if avg_worsening:
        scores["PI"] -= 2.0
    if low_diversity:
        scores["PI"] -= 1.0

    # ---- Bandit prior shrinkage --------------------------------------------------------
    # The heuristic votes above are a hand-built PRIOR over which AF suits
    # the situation. In bandit mode they are kept but DECAYED as reward
    # evidence accumulates, so early batches (no reward data) follow the
    # prior and later batches follow the data -- a pseudo-count shrinkage
    # with prior_weight = tau / (tau + n_eff). This beats both the static
    # heuristic (never adapts; loses under non-stationarity) and the
    # prior-free pure bandit (no cold-start prior). bandit_pure=True is the
    # tau=0 limit. The hard PI pre-gate ban is applied AFTER scaling at
    # full strength so it remains a real safety constraint, not a vote.
    if config.enable_bandit:
        tau = 0.0 if config.bandit_pure else config.bandit_prior_strength
        n_eff = max(n_batches - 1, 0)          # batches carrying reward signal
        prior_weight = tau / (tau + n_eff) if (tau + n_eff) > 0 else 0.0
        scores = {af: prior_weight * s for af, s in scores.items()}

    if pi_banned:
        scores["PI"] -= 3.0

    credits: Dict[str, float] = {}
    if config.enable_credit:
        credits = _acquisition_credit(batch_summary, tolerance, config)
        for af, c in credits.items():
            scores[af] += config.w_credit * c

    # ---- Usage balancing (soft portfolio pressure) ----------------------------------------
    # In bandit mode the heuristic usage-share term is replaced by the
    # discounted-UCB exploration bonus (same role, principled scaling).
    balance_bonuses: Dict[str, float] = {}
    usage_shares: Dict[str, float] = {}
    bandit_bonuses: Dict[str, float] = {}
    if config.enable_bandit:
        _, usage_shares = _usage_balance(batch_summary, config)  # diagnostics only
        bandit_bonuses = _bandit_exploration_bonus(batch_summary, config)
        for af, b in bandit_bonuses.items():
            scores[af] += b
    elif config.w_balance > 0:
        balance_bonuses, usage_shares = _usage_balance(batch_summary, config)
        for af, b in balance_bonuses.items():
            scores[af] += b

    # ---- Budget shaping -------------------------------------------------------------------
    if progress is not None:
        if endgame:
            t = (progress - (1.0 - config.endgame_fraction)) / config.endgame_fraction
            t = float(np.clip(t, 0.0, 1.0))
            scores["PI"] += config.w_budget * t
            scores["UCB"] -= config.w_budget * t
        elif early_phase:
            t = (config.early_fraction - progress) / config.early_fraction
            t = float(np.clip(t, 0.0, 1.0))
            scores["UCB"] += config.w_budget * t
            scores["PI"] -= config.w_budget * t

    # ---- Convergence nudge -----------------------------------------------------------------
    # If the model expects no further improvement (max_ei below tolerance)
    # and we are NOT stuck in a sub-optimal basin, the exploration phase is
    # statistically over: tip the balance toward exploitation and away from
    # exploration. This is a soft score nudge (the rescue ladder and guards
    # still apply); the hard escape case is handled earlier in the ladder.
    if model_converged and not stuck_in_local_opt:
        scores["PI"] += config.w_uncertainty
        scores["UCB"] -= config.w_uncertainty

    warnings = extra_warnings + _data_warnings(
        batch_summary, yrange, mean_uncertainty, diversity
    )

    diagnostics = {
        "n_completed_batches": n_batches,
        "current_acquisition": current_acquisition,
        "best_yield": best_yield,
        "reference_best": ref_best,
        "batch_average_yield": batch_avg,
        "delta_best": delta_best,
        "delta_batch_average": delta_avg,
        "avg_slope": avg_slope,
        "best_slope": best_slope,
        "adaptive_yield_tolerance": tolerance,
        "avg_drop_tolerance": avg_drop_tol,
        "observed_yield_range": yrange,
        "robust_statistics_active": robust_active,
        "latest_hit_rate": latest_hit_rate,
        "recent_hit_rate": recent_hit_rate,
        "latest_hit_rate_lower_bound": latest_hit_lcb,
        "recent_hit_rate_lower_bound": recent_hit_lcb,
        "latest_hit_n_trials": latest_hit_n,
        "recent_hit_n_trials": recent_hit_n,
        "record_rate_deficit": record_deficit,
        **record_info,
        "hit_threshold": hit_threshold,
        "stagnated": stagnated,
        "avg_worsening": avg_worsening,
        "stagnated_run_rule": stagnated_run_rule,
        "avg_worsening_run_rule": avg_worsening_run_rule,
        "stagnation_evidence_wealth": stagnation_wealth,
        "worsening_evidence_wealth": worsening_wealth,
        "yield_dispersion_ratio": dispersion_ratio,
        "distribution_drift_w1": w1_drift,
        "diversity_proxy_active": diversity_proxy_active,
        "improving": improving,
        "high_batch_average": high_batch_average,
        "mean_uncertainty": mean_uncertainty,
        "diversity": diversity,
        "high_uncertainty": high_uncertainty,
        "low_uncertainty": low_uncertainty,
        "low_diversity": low_diversity,
        "unknown_uncertainty": unknown_uncertainty,
        "unknown_diversity": unknown_diversity,
        "batches_since_switch": batches_since_switch,
        "acquisition_credits": credits,
        "usage_shares": usage_shares,
        "balance_bonuses": balance_bonuses,
        "bandit_bonuses": bandit_bonuses,
        "batches_remaining": batches_remaining,
        "budget_progress": progress,
        "endgame": endgame,
        "early_phase": early_phase,
    }

    def _result(next_acq: str, reason: str, state: str) -> Dict[str, Any]:
        return {
            "next_acquisition": next_acq,
            "reason": reason,
            "state": state,
            "diagnostics": diagnostics,
            "scores": scores,
            "warnings": warnings,
            "batch_summary": batch_summary,
        }

    # In the endgame there is no budget left to recover from a wide
    # exploration detour, so rescues redirect to EI instead of UCB.
    explore_target = "EI" if endgame else "UCB"

    previous_acquisition = _previous_acquisition(batch_summary, current_acquisition)
    severe_drop = delta_avg <= -config.revert_drop_multiplier * avg_drop_tol
    best_failed_to_improve = delta_best < tolerance
    diagnostics["previous_acquisition"] = previous_acquisition
    diagnostics["severe_drop"] = severe_drop
    diagnostics["max_ei"] = max_ei
    diagnostics["optimism_gap"] = optimism_gap
    diagnostics["model_converged"] = model_converged
    diagnostics["stuck_in_local_opt"] = stuck_in_local_opt
    diagnostics["prolonged_stall"] = prolonged_stall

    # ---- Explicit rules -----------------------------------------------------------------
    # A switch is a "mistake" only if the batch average collapsed AND the
    # incumbent best did not improve. On real landscapes a new acquisition
    # function's first batch often explores, lowering the batch AVERAGE even
    # while it holds or advances the best-so-far -- exploration working, not
    # a mistake. Requiring the best to also stall stops the revert from
    # yanking a productive switch straight back (the failure mode that kept
    # PI/EI from ever stabilizing in the real run).
    if (
        config.enable_mistake_revert
        and batches_since_switch == 1
        and previous_acquisition in BO_ACQUISITIONS
        and severe_drop
        and best_failed_to_improve
    ):
        proposed, state = previous_acquisition, "revert_mistake"
        reason = (
            f"Revert {current_acquisition} → {previous_acquisition}: first batch "
            f"after switching dropped the average (Δavg {delta_avg:.3g} ≤ "
            f"-{config.revert_drop_multiplier:g}× tol {avg_drop_tol:.3g}) AND did "
            f"not improve the best (Δbest {delta_best:.3g}); treating as a mistake."
        )
    elif stuck_in_local_opt and current_acquisition != "UCB":
        # Yield has plateaued but the model still believes substantially
        # better yields are reachable elsewhere => sub-optimal basin.
        # Escape with exploration (or a Sobol restart in severe cases).
        escape_to = "EI" if endgame else "UCB"
        proposed, state = escape_to, "escape_local_optimum"
        reason = (
            f"Escape sub-optimal basin → {escape_to}: yield has plateaued, but "
            f"the model's optimism gap ({optimism_gap:.3g}) exceeds "
            f"{config.escape_gap_mult:g}× tolerance ({tolerance:.3g}) — it "
            f"believes better yields exist in unexplored space, so break out "
            f"with exploration rather than keep exploiting a local optimum."
        )
    elif (
        config.allow_sobol_restart
        and not endgame
        and current_acquisition == "UCB"
        and stuck_in_local_opt
    ):
        proposed, state = "Sobol", "random_restart"
        reason = (
            "Random restart: even UCB exploration is stuck in a sub-optimal "
            "basin while the model still expects better elsewhere; inject a "
            "fresh Sobol batch to escape."
        )
    elif (
        config.enable_stall_escape
        and prolonged_stall
        and not endgame
        and current_acquisition != "UCB"
    ):
        # Model-free escape: best-so-far flat for a long run. We cannot tell
        # converged-from-stuck without model_signal, but forcing exploration
        # is the right move either way after a prolonged plateau (escapes a
        # basin if stuck; costs one batch if already converged). This is the
        # campaign-6 fix when no model signal is supplied.
        proposed, state = "UCB", "escape_prolonged_stall"
        reason = (
            f"Escape prolonged stall → UCB: best yield has not improved for "
            f"{config.stall_escape_patience}+ batches while exploiting; force "
            f"exploration to break out of a possible sub-optimal region."
        )
    elif (
        config.allow_sobol_restart
        and not endgame
        and current_acquisition == "UCB"
        and stagnated
        and batches_since_switch >= config.exploration_patience
        and record_deficit is not False  # restart only when records are at
        # or below the random-sampling baseline (or the test is inconclusive)
    ):
        proposed, state = "Sobol", "random_restart"
        reason = (
            f"Random restart: UCB has stagnated for {batches_since_switch} "
            f"batches and new bests are arriving no faster than random "
            f"sampling would produce; the surrogate is likely misled "
            f"everywhere it has data, so run one fresh Sobol batch."
        )
    elif current_acquisition == "PI" and stagnated:
        proposed, state = "EI", "switch_back_soft"
        reason = "Switch PI → EI: exploitation plateau detected over the patience window."
    elif current_acquisition == "EI" and stagnated and (
        high_uncertainty or low_diversity or avg_worsening
    ):
        proposed, state = explore_target, "switch_back_explore"
        reason = (
            f"Switch EI → {explore_target}: balanced search has plateaued and "
            f"an exploration signal is present"
            + (" (endgame: staying budget-efficient with EI)." if endgame else ".")
        )
    elif stagnated and (low_diversity or avg_worsening):
        proposed, state = explore_target, "rescue_exploration"
        reason = (
            f"Switch to {explore_target}: stagnation plus diversity collapse "
            f"or average-yield decline"
            + (" (endgame: staying budget-efficient with EI)." if endgame else ".")
        )
    else:
        proposed = max(scores, key=scores.get)
        if proposed == "UCB":
            state, reason = "explore", "Use UCB: exploration/rescue score is highest."
        elif proposed == "EI":
            state, reason = "balance", "Use EI: balanced improvement score is highest."
        else:
            state, reason = "exploit", (
                "Use PI: low uncertainty and high-yield enrichment justify exploitation."
            )

        # Scheduled rotation (portfolio behavior): when enabled and the
        # current AF has served its minimum hold, move to the least-used BO
        # acquisition, subject to safety guards. Rescue rules above always
        # take precedence; the endgame disables rotation (commit, don't tour).
        if (
            config.enable_rotation
            and not endgame
            and current_acquisition in BO_ACQUISITIONS
            and batches_since_switch >= config.min_batches_per_acquisition
        ):
            ordered = sorted(
                BO_ACQUISITIONS,
                key=lambda af: (usage_shares.get(af, 0.0), af != current_acquisition),
            )
            for candidate in ordered:
                if candidate == current_acquisition:
                    continue
                if candidate == "PI" and (
                    n_batches < config.min_batches_before_PI
                    or high_uncertainty
                    or avg_worsening
                ):
                    continue
                proposed, state = candidate, "rotate"
                reason = (
                    f"Rotate {current_acquisition} → {candidate}: scheduled "
                    f"balanced rotation (least-used acquisition in the last "
                    f"{config.balance_window} batches)."
                )
                break

    # Sobol must never be held once adaptive switching is active; a proposed
    # random restart likewise bypasses hold and hysteresis.
    leaving_sobol = current_acquisition == "Sobol"
    restart_proposed = proposed == "Sobol"
    revert_proposed = state == "revert_mistake"
    escape_proposed = state in ("escape_local_optimum", "escape_prolonged_stall")

    # ---- Anti-bouncing rule ------------------------------------------------------------------
    if (
        proposed != current_acquisition
        and not leaving_sobol
        and not restart_proposed
        and not revert_proposed
        and not escape_proposed   # escaping a confirmed sub-optimal basin is urgent
        and batches_since_switch < config.min_batches_per_acquisition
    ):
        serious_rescue = stagnated and (avg_worsening or low_diversity)
        pi_plateau = current_acquisition == "PI" and stagnated
        if not serious_rescue and not pi_plateau:
            return _result(
                current_acquisition,
                f"Hold {current_acquisition}: minimum hold period not reached "
                f"({batches_since_switch}/{config.min_batches_per_acquisition}).",
                "hold_minimum",
            )

    # ---- Hysteresis rule -----------------------------------------------------------------------
    if (
        proposed != current_acquisition
        and not leaving_sobol
        and not restart_proposed
        and state in {"explore", "balance", "exploit"}
    ):
        current_score = scores.get(current_acquisition, 0.0)
        proposed_score = scores.get(proposed, 0.0)
        if proposed_score - current_score < config.switch_score_margin:
            return _result(
                current_acquisition,
                f"Hold {current_acquisition}: proposed switch to {proposed} "
                f"does not exceed hysteresis margin "
                f"({proposed_score - current_score:.2f} < {config.switch_score_margin:.2f}).",
                "hold_hysteresis",
            )

    return _result(proposed, reason, state)


# Backwards-compatible alias for existing code.
choose_next_acquisition_v2 = choose_next_acquisition


# --------------------------------------------------------------------------- #
# Auto-calibration
# --------------------------------------------------------------------------- #
def suggest_config(
    history: pd.DataFrame,
    batch_col: str = "batch",
    yield_col: str = "yield",
    acq_col: str = "acquisition",
    planned_total_batches: Optional[int] = None,
    search_space_size: Optional[int] = None,
) -> Tuple[SwitchConfig, Dict[str, Any]]:
    """
    Inspect a dataset and return a SwitchConfig tuned to it, plus a rationale.

    Heuristics (a starting point, not gospel):
    - Small batches (median < 4 trials) -> more patience before switching.
    - High batch-to-batch noise relative to the yield range (ROUGH data) ->
      larger noise_multiplier, more patience, AND stronger balance pressure
      (signals are less trustworthy, so balanced AF rotation is safer).
    - Very low noise (SMOOTH data) -> weaker balance pressure (signals are
      trustworthy; let the data-driven scores commit to the best AF).
    - Outlier signature (max far above the 95th percentile) -> robust=True.
    - Short planned campaigns -> allow PI earlier; long ones -> later.
    - Small search spaces (FEW combinations) -> exploit earlier; huge search
      spaces (MANY combinations) -> hold off exploitation and keep exploring.
    """
    bs = summarize_batches(history, batch_col, yield_col, acq_col)
    rationale: Dict[str, Any] = {}

    yvals = pd.to_numeric(history[yield_col], errors="coerce").dropna().to_numpy(dtype=float)
    yrange = float(yvals.max() - yvals.min()) if len(yvals) else 0.0
    n_trials = len(yvals)

    median_batch_size = float(bs["n_experiments"].median())
    rationale["median_batch_size"] = median_batch_size

    deltas = bs["delta_batch_average"].iloc[1:]
    noise = _robust_noise(deltas.to_numpy(dtype=float)) if len(deltas) >= 2 else 0.0
    noise_ratio = noise / yrange if yrange > 0 else 0.0
    rationale["batch_delta_noise"] = noise
    rationale["noise_to_range_ratio"] = noise_ratio

    # Outlier signature: best far above the bulk of the data.
    outlier = False
    if len(yvals) >= 10 and yrange > 0:
        q95 = float(np.quantile(yvals, 0.95))
        outlier = (float(yvals.max()) - q95) > 0.15 * yrange
    rationale["outlier_detected"] = outlier

    patience = 2
    rolling_window = 3
    noise_multiplier = 0.5
    w_balance = 0.5  # default mild rotation pressure

    if median_batch_size < 4:
        patience = 3
        rolling_window = 4
        rationale["patience_note"] = (
            "Small batches (<4 trials): increased patience/window to avoid "
            "switching on batch-level noise."
        )

    if noise_ratio > 0.10:
        patience = max(patience, 3)
        rolling_window = max(rolling_window, patience + 1)
        noise_multiplier = 0.75
        w_balance = 1.5
        rationale["noise_note"] = (
            "ROUGH data (batch-to-batch variation > 10% of yield range): "
            "raised noise_multiplier and patience, and strengthened balanced "
            "AF rotation since individual signals are less trustworthy."
        )
    elif 0.0 < noise_ratio < 0.03:
        w_balance = 0.25
        rationale["noise_note"] = (
            "SMOOTH data (variation < 3% of yield range): signals are "
            "trustworthy, so balance pressure is reduced and the data-driven "
            "scores are allowed to commit to whichever AF is delivering."
        )

    min_batches_before_pi = 4
    if planned_total_batches is not None and planned_total_batches > 0:
        min_batches_before_pi = int(
            min(max(3, math.ceil(0.3 * planned_total_batches)), 8)
        )
        rationale["pi_note"] = (
            f"PI allowed after ~30% of the planned {planned_total_batches}-batch "
            f"budget (>= {min_batches_before_pi} batches)."
        )

    if search_space_size is not None and search_space_size > 0:
        coverage = n_trials / search_space_size
        rationale["search_space_coverage"] = coverage
        if search_space_size <= 200 or coverage >= 0.10:
            min_batches_before_pi = min(min_batches_before_pi, 3)
            rationale["space_note"] = (
                f"FEW combinations ({search_space_size}; coverage "
                f"{coverage:.1%}): the space is quickly mapped, so "
                f"exploitation is allowed earlier."
            )
        elif search_space_size >= 10_000 and coverage < 0.01:
            min_batches_before_pi = max(min_batches_before_pi, 5)
            rationale["space_note"] = (
                f"MANY combinations ({search_space_size}; coverage "
                f"{coverage:.2%}): held off exploitation to keep exploring "
                f"a barely-sampled space."
            )

    cfg = SwitchConfig(
        patience=patience,
        rolling_window=rolling_window,
        noise_multiplier=noise_multiplier,
        robust=outlier,
        min_batches_before_PI=min_batches_before_pi,
        min_completed_batches_before_switching=max(3, patience),
        w_balance=w_balance,
    )
    if outlier:
        rationale["robust_note"] = (
            "Best yield sits far above the 95th percentile: enabled robust "
            "statistics so one outlier cannot distort thresholds."
        )
    return cfg, rationale


def format_decision(decision: Dict[str, Any]) -> str:
    """Pretty one-line decision string for logging."""
    d = decision["diagnostics"]
    scores = decision["scores"]
    line = (
        f"Next AF: {decision['next_acquisition']} | "
        f"State: {decision['state']} | "
        f"Reason: {decision['reason']} | "
        f"Best: {d.get('best_yield', np.nan):.3g} | "
        f"Batch avg: {d.get('batch_average_yield', np.nan):.3g} | "
        f"Scores UCB/EI/PI: "
        f"{scores.get('UCB', np.nan):.2f}/"
        f"{scores.get('EI', np.nan):.2f}/"
        f"{scores.get('PI', np.nan):.2f}"
    )
    if decision.get("warnings"):
        line += f" | WARNINGS: {len(decision['warnings'])}"
    return line
