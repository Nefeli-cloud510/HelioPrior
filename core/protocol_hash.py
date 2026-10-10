from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "research-engine/2.0"


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"protocol_hash 不支持的类型: {type(value).__name__}")


def protocol_hash(obj: Any) -> str:
    """Canonical SHA-256 fingerprint of a protocol/configuration object.

    Any change to the execution-relevant protocol parameters (features,
    horizon, alpha grid, split ratio, ...) changes the hash, so results can be
    bound deterministically to a protocol snapshot.
    """

    canonical = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
