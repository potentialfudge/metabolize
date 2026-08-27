"""
Translation layer between the app's parameter config (as filled in via the
setup form and stored in campaigns.config) and BayBE's SearchSpace / campaign
objects, plus the glue that turns the meta blender's decide_blend() output
into an actual next-batch recommendation.

Nothing in ui/ should import BayBE directly -- everything BayBE-specific lives
here, so the UI layer only ever deals with plain dicts/DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from baybe.parameters import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    CategoricalParameter,
    SubstanceParameter,
)
from baybe.searchspace import SearchSpace

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from meta_blender import decide_blend, blend_scores, BlendConfig


# --------------------------------------------------------------------------- #
# Encoding auto-selection (simple mode)
# --------------------------------------------------------------------------- #
def _looks_like_smiles(values: List[str]) -> bool:
    """Crude heuristic: SMILES strings are dense in bracket/bond symbols and
    don't look like plain words. Good enough as a simple-mode default; advanced
    mode lets the user override this entirely."""
    try:
        from rdkit import Chem
    except ImportError:
        return False
    hits = 0
    for v in values[:5]:  # sample a few, not the whole list
        if Chem.MolFromSmiles(str(v)) is not None:
            hits += 1
    return hits >= max(1, len(values[:5]) // 2)


def _auto_encoding(param: Dict[str, Any]) -> str:
    if param["type"] == "continuous":
        return "none"
    if _looks_like_smiles(param.get("values", [])):
        return "smiles"
    return "OHE"


# --------------------------------------------------------------------------- #
# config["parameters"] -> BayBE parameter objects
# --------------------------------------------------------------------------- #
def _build_baybe_parameter(param: Dict[str, Any]):
    name = param["name"]
    ptype = param["type"]
    encoding = param.get("encoding") or _auto_encoding(param)

    if ptype == "categorical":
        values = [str(v) for v in param["values"]]
        if encoding == "smiles":
            # SubstanceParameter expects {label: smiles} -- here label == smiles
            # unless the user's values are already named compounds elsewhere.
            return SubstanceParameter(name=name, data={v: v for v in values}, encoding="MORDRED")
        return CategoricalParameter(name=name, values=values, encoding="OHE")

    if ptype == "continuous":
        lo, hi = float(param["min"]), float(param["max"])
        interval = param.get("interval")
        if interval:
            n_steps = int(round((hi - lo) / float(interval))) + 1
            values = [round(lo + i * float(interval), 10) for i in range(n_steps)]
            return NumericalDiscreteParameter(name=name, values=values)
        return NumericalContinuousParameter(name=name, bounds=(lo, hi))

    raise ValueError(f"Unknown parameter type: {ptype!r}")


def build_search_space(config: Dict[str, Any]) -> SearchSpace:
    """Turn campaigns.config['parameters'] into a BayBE SearchSpace."""
    params = [_build_baybe_parameter(p) for p in config["parameters"]]
    return SearchSpace.from_product(parameters=params)


# --------------------------------------------------------------------------- #
# Init set generation
# --------------------------------------------------------------------------- #
def _pick_from_list(u: np.ndarray, choices: List) -> List:
    """Map Sobol points u in [0,1) onto indices into `choices`."""
    idx = np.floor(u * len(choices)).astype(int)
    idx = np.clip(idx, 0, len(choices) - 1)
    return [choices[i] for i in idx]


def generate_sobol_init(config: Dict[str, Any], seed: int) -> pd.DataFrame:
    """
    Generate the initial batch via a genuine Sobol sequence (scipy.stats.qmc),
    scaled/indexed per parameter type. This is independent of BayBE's
    recommenders -- BayBE's RandomRecommender is plain uniform random, not a
    low-discrepancy sequence, and clusters more than we want for an init set.

    - continuous: Sobol u scaled into [min, max]
    - discrete (has `interval`) / categorical: Sobol u used to index into the
      allowed value list
    """
    from scipy.stats import qmc

    params = config["parameters"]
    n = config["init_size"]
    if n <= 0:
        raise ValueError("init_size must be > 0")
    if not params:
        raise ValueError("No parameters provided")

    eng = qmc.Sobol(d=len(params), scramble=True, seed=seed)
    u = eng.random(n)

    cols: Dict[str, Any] = {}
    for j, p in enumerate(params):
        uj = u[:, j]
        if p["type"] == "continuous" and not p.get("interval"):
            lo, hi = float(p["min"]), float(p["max"])
            cols[p["name"]] = lo + uj * (hi - lo)
        elif p["type"] == "continuous" and p.get("interval"):
            lo, hi, interval = float(p["min"]), float(p["max"]), float(p["interval"])
            n_steps = int(round((hi - lo) / interval)) + 1
            values = [round(lo + i * interval, 10) for i in range(n_steps)]
            cols[p["name"]] = _pick_from_list(uj, values)
        elif p["type"] == "categorical":
            cols[p["name"]] = _pick_from_list(uj, list(p["values"]))
        else:
            raise ValueError(f"Unsupported parameter type for Sobol init: {p['type']!r}")

    return pd.DataFrame(cols)


def validate_uploaded_init(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """Check an uploaded init CSV has the right columns. Returns a list of
    problems (empty list = valid)."""
    problems = []
    expected = {p["name"] for p in config["parameters"]}
    got = set(df.columns) - {config["target_name"]}
    missing = expected - got
    extra = got - expected
    if missing:
        problems.append(f"Missing columns: {sorted(missing)}")
    if extra:
        problems.append(f"Unexpected columns: {sorted(extra)}")
    return problems


# --------------------------------------------------------------------------- #
# Encoding for the GP surrogate (separate from BayBE's SearchSpace encodings --
# this is a plain, consistent one-hot + passthrough scheme used only to fit
# the raw BoTorch GP below, mirroring the approach already proven in the
# benchmarking pipeline's schsf_meta_aug_runner.py)
# --------------------------------------------------------------------------- #
def _encode_matrix(df: pd.DataFrame, parameters: List[Dict[str, Any]]) -> np.ndarray:
    """Encode a dataframe of raw parameter values into a numeric matrix.
    Categorical -> one-hot against the parameter's DECLARED value list (not
    just what's present in df), so history and candidate encodings always
    line up column-for-column even if not every category appears yet."""
    cols = []
    for p in parameters:
        if p["type"] == "categorical":
            for v in p["values"]:
                cols.append((df[p["name"]].astype(str) == str(v)).astype(float).to_numpy())
        else:
            cols.append(df[p["name"]].astype(float).to_numpy())
    return np.column_stack(cols)


# --------------------------------------------------------------------------- #
# BoTorch surrogate (fit directly, bypassing BayBE's Campaign/Recommender --
# this mirrors the proven pattern from schsf_meta_aug_runner.py so we get
# real per-acquisition UCB/EI/PI scores to feed blend_scores())
# --------------------------------------------------------------------------- #
def _fit_gp(X_train: np.ndarray, y_train: np.ndarray):
    import torch
    import gpytorch
    from gpytorch.priors import GammaPrior
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    dtype = torch.double
    tx = torch.as_tensor(X_train, dtype=dtype)
    ty = torch.as_tensor(y_train, dtype=dtype).unsqueeze(-1)
    y_mean = ty.mean()
    y_std = ty.std().clamp_min(1e-6)
    ty_s = (ty - y_mean) / y_std

    model = SingleTaskGP(
        tx, ty_s,
        covar_module=gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, lengthscale_prior=GammaPrior(3.0, 6.0))
        ),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model, float(y_mean), float(y_std)


def _posterior_mu_sigma(model, y_mean: float, y_std: float, X_cand: np.ndarray):
    import torch
    tx = torch.as_tensor(X_cand, dtype=torch.double)
    model.eval()
    with torch.no_grad():
        post = model.posterior(tx)
        mu_s = post.mean.squeeze(-1)
        sd_s = post.variance.clamp_min(1e-12).sqrt().squeeze(-1)
    mu = (mu_s * y_std + y_mean).cpu().numpy()
    sd = (sd_s * y_std).cpu().numpy()
    return mu.astype(float), sd.astype(float)


def _acquisition_values(mu: np.ndarray, sigma: np.ndarray, y_best: float, beta: float, xi: float):
    """Analytic UCB, EI, PI -- same formulas as schsf_meta_aug_runner.py."""
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-9)
    ucb = mu + beta * sigma
    imp = mu - y_best - xi
    z = imp / sigma
    ei = np.maximum(imp * norm.cdf(z) + sigma * norm.pdf(z), 0.0)
    pi = norm.cdf(z)
    return ucb, ei, pi


def _sample_candidate_pool(config: Dict[str, Any], n: int, seed: int) -> pd.DataFrame:
    """Reuse the Sobol sampling logic to generate a large candidate pool to
    score and rank, rather than enumerating the full (possibly huge) space."""
    pool_config = dict(config)
    pool_config["init_size"] = n
    return generate_sobol_init(pool_config, seed=seed)


def _annealed_xi(round_num: int, total_rounds: int, xi_max: float, xi_min: float) -> float:
    """Decay xi linearly from xi_max (round 0, favors more exploration in EI)
    to xi_min (final round, favors pure exploitation) -- same schedule used in
    the benchmarking pipeline's schsf_meta_aug_runner.py."""
    if total_rounds <= 0:
        return xi_min
    frac = float(np.clip(round_num / total_rounds, 0.0, 1.0))
    return xi_max * (1 - frac) + xi_min * frac


# --------------------------------------------------------------------------- #
# Next-batch recommendation, driven by the meta blender
# --------------------------------------------------------------------------- #
def recommend_next_batch(
    config: Dict[str, Any],
    history_df: pd.DataFrame,
    batch_size: int,
    total_batches: Optional[int] = None,
    candidate_pool_size: int = 500,
    candidate_seed: int = 0,
    ei_xi_max: float = 0.01,
    ei_xi_min: float = 0.001,
) -> Dict[str, Any]:
    """
    Fits a GP surrogate on ingested history, computes UCB/EI/PI for a sampled
    candidate pool, blends them using decide_blend()'s weights, and returns
    the top batch_size candidates plus the full decide_blend() output (for
    the stats page).

    history_df must have one row per experiment with parameter columns, a
    'batch' column (round number), and a column named config['target_name'].

    ei_xi_max / ei_xi_min: EI's exploration margin decays linearly from
    ei_xi_max at round 0 to ei_xi_min at the final round (see _annealed_xi),
    matching the schedule already validated in the benchmarking pipeline.
    """
    params = config["parameters"]
    target_col = config["target_name"]

    blend_history = history_df.rename(columns={target_col: "yield"})
    if "batch" not in blend_history.columns:
        raise ValueError("history_df must include a 'batch' column (round number)")

    current_round = int(blend_history["batch"].max()) + 1  # round we're generating

    # ---- 1. meta blend decision (weights, beta) ----
    blend_out = decide_blend(
        blend_history,
        batch_col="batch",
        yield_col="yield",
        total_batches=total_batches,
        config=BlendConfig(),
    )
    weights = blend_out["weights"]
    beta = blend_out["beta"]

    # ---- 2. fit GP on ingested history ----
    X_train = _encode_matrix(history_df, params)
    y_train = history_df[target_col].to_numpy(float)
    model, y_mean, y_std = _fit_gp(X_train, y_train)
    y_best = float(y_train.max())

    # ---- 3. sample + score a candidate pool ----
    candidates = _sample_candidate_pool(config, candidate_pool_size, seed=candidate_seed)
    param_names = [p["name"] for p in params]

    # Dedupe: with a small search space (few categorical options and/or a
    # coarse continuous interval), a large sampled pool can contain many
    # repeats of the same point. Duplicate inputs get identical GP scores,
    # so without deduping, top-k selection can return the same point
    # multiple times instead of batch_size distinct recommendations.
    candidates = candidates.drop_duplicates(subset=param_names).reset_index(drop=True)

    # Exclude points already measured in any prior round -- re-recommending
    # an already-run experiment wastes budget and never adds new information.
    already_measured = history_df[param_names].astype(str)
    already_measured_set = set(map(tuple, already_measured.to_numpy()))
    candidates_str = candidates[param_names].astype(str)
    is_new = ~candidates_str.apply(tuple, axis=1).isin(already_measured_set)
    candidates = candidates[is_new].reset_index(drop=True)

    effective_batch_size = min(batch_size, len(candidates))
    if effective_batch_size < batch_size:
        # Search space itself (minus already-measured points) is smaller
        # than the requested batch -- a real constraint, not a bug. Return
        # everything unique and unmeasured that's left.
        pass

    X_cand = _encode_matrix(candidates, params)
    mu, sigma = _posterior_mu_sigma(model, y_mean, y_std, X_cand)

    xi = _annealed_xi(current_round, total_batches or 0, ei_xi_max, ei_xi_min)
    ucb, ei, pi = _acquisition_values(mu, sigma, y_best, beta=beta, xi=xi)

    # ---- 4. blend + select top batch_size ----
    blended = blend_scores(ucb, ei, pi, weights)
    top_idx = np.argsort(blended)[::-1][:effective_batch_size]
    chosen = candidates.iloc[top_idx].reset_index(drop=True)
    chosen["predicted_mean"] = mu[top_idx]
    chosen["predicted_std"] = sigma[top_idx]

    return {
        "blend": blend_out,
        "recommendations": chosen,
        "xi": xi,
        "requested_batch_size": batch_size,
        "actual_batch_size": effective_batch_size,
    }