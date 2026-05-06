"""Lightweight Comet wrapper that no-ops when no API key is available.

The training loop calls :func:`log_metrics` and :func:`log_image`
unconditionally; the helpers degrade to no-ops when ``experiment`` is
``None`` so callers don't need to guard every logging site.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

from config import Config, flatten_for_comet

logger = logging.getLogger(__name__)


def _try_build_experiment(cfg: Config):
    """Try to instantiate a Comet ``Experiment`` based on env+config state."""
    if not cfg.comet.enabled:
        return None
    if not os.environ.get("COMET_API_KEY"):
        logger.info("Comet disabled: COMET_API_KEY not set")
        return None
    try:
        from comet_ml import Experiment  # type: ignore[import-not-found]
    except ImportError as e:
        logger.warning("Comet logging disabled (failed to import comet_ml: %s)", e)
        return None
    try:
        exp = Experiment(workspace=cfg.comet.workspace, project_name=cfg.comet.project)
    except Exception as e:  # pragma: no cover - depends on user env
        logger.warning("Comet Experiment init failed (%s); continuing without Comet", e)
        return None
    return exp


def build_experiment(cfg: Config):
    """Public factory: returns an active ``Experiment`` or ``None``."""
    exp = _try_build_experiment(cfg)
    if exp is None:
        return None
    exp.set_name(cfg.run_name)
    try:
        params = flatten_for_comet(cfg)
        exp.log_parameters(params)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Comet log_parameters failed: %s", e)
    return exp


def log_metrics(experiment: Any, metrics: dict[str, float], step: int) -> None:
    """Log finite scalar metrics; drop NaN/non-numeric entries silently."""
    if experiment is None or not metrics:
        return
    clean: dict[str, float] = {}
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            continue
        clean[k] = fv
    if clean:
        try:
            experiment.log_metrics(clean, step=step)
        except Exception as e:  # pragma: no cover
            logger.warning("Comet log_metrics failed: %s", e)


def log_image(experiment: Any, image_path: str, name: str, step: int) -> None:
    """Upload a saved PNG to Comet under the given ``name``."""
    if experiment is None:
        return
    try:
        experiment.log_image(image_path, name=name, step=step)
    except Exception as e:  # pragma: no cover
        logger.warning("Comet log_image failed: %s", e)


def end(experiment: Any) -> None:
    """Best-effort end of the Comet experiment."""
    if experiment is None:
        return
    try:
        experiment.end()
    except Exception:  # pragma: no cover
        pass


__all__ = ["build_experiment", "end", "log_image", "log_metrics"]
