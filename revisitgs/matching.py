"""Deterministic collision resolution for sparse Gaussian transport."""

from __future__ import annotations

import ctypes
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "native/greedy_match.c"
BUILD = Path(tempfile.gettempdir()) / "revisitgs_native"
LIBRARY = BUILD / ("libgreedy_" + hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12] + ".so")
_LIB = None


def _compile() -> Path:
    if not LIBRARY.is_file():
        BUILD.mkdir(parents=True, exist_ok=True)
        # Concurrent cohort workers must not compile into the same temporary file.
        with tempfile.NamedTemporaryFile(dir=BUILD, suffix=".so", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            subprocess.run(
                ["cc", "-O3", "-shared", "-fPIC", str(SOURCE), "-o", str(temporary)],
                check=True,
            )
            temporary.replace(LIBRARY)
        finally:
            temporary.unlink(missing_ok=True)
    return LIBRARY


def _library():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(_compile()))
        i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS")
        _LIB.greedy_match.argtypes = [i64, i64, i64, ctypes.c_int64, u8, u8, i64]
        _LIB.greedy_match.restype = None
    return _LIB


def resolve(
    order: np.ndarray,
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    source_count: int,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.ascontiguousarray(order, dtype=np.int64)
    edge_source = np.ascontiguousarray(edge_source, dtype=np.int64)
    edge_target = np.ascontiguousarray(edge_target, dtype=np.int64)
    source_used = np.zeros(source_count, dtype=np.uint8)
    target_used = np.zeros(target_count, dtype=np.uint8)
    assignment = np.full(source_count, -1, dtype=np.int64)
    _library().greedy_match(
        order, edge_source, edge_target, len(order), source_used, target_used, assignment
    )
    return source_used.astype(bool), target_used.astype(bool), assignment
