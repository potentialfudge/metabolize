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
# Next-batch recommendation, driven by the meta blender
# --------------------------------------------------------------------------- #
def recommend_next_batch(
    config: Dict[str, Any],
    history_df: pd.DataFrame,
    batch_size: int,
    total_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Runs decide_blend() on the ingested history, scores a pool of candidates
    from the search space using the returned weights, and returns the top
    batch_size candidates plus the full decide_blend() output (for the stats
    page).

    history_df must have one row per experiment with parameter columns plus
    a column named config['target_name'].
    """
    from baybe.recommenders import RandomRecommender

    target_col = config["target_name"]
    blend_history = history_df.rename(columns={target_col: "yield"})
    if "batch" not in blend_history.columns:
        raise ValueError("history_df must include a 'batch' column (round number)")

    blend_out = decide_blend(
        blend_history,
        batch_col="batch",
        yield_col="yield",
        total_batches=total_batches,
        config=BlendConfig(),
    )

    search_space = build_search_space(config)
    # Pull a larger candidate pool than we need, then rank with blend_scores.
    pool_size = max(batch_size * 20, 200)
    candidates = RandomRecommender().recommend(batch_size=pool_size, searchspace=search_space)

    # NOTE: scoring candidates with actual UCB/EI/PI acquisition values requires
    # a fitted BayBE surrogate/campaign object bound to history_df. This wiring
    # (fit surrogate on history_df -> get per-acquisition scores for candidates
    # -> blend_scores()) is the next piece to fill in once a BayBE Campaign is
    # constructed in the UI layer; recommend_next_batch currently returns the
    # blend decision + a candidate pool so that wiring has a clear seam.
    return {
        "blend": blend_out,
        "candidates": candidates.reset_index(drop=True),
    }