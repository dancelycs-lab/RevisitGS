"""Quantization and container utilities for the RevisitGS revision stream."""

from __future__ import annotations

import io
import lzma

import numpy as np


def serialize(arrays: dict[str, np.ndarray]) -> bytes:
    raw = io.BytesIO()
    np.savez(raw, **arrays)
    return lzma.compress(raw.getvalue(), preset=9 | lzma.PRESET_EXTREME)


def deserialize(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(lzma.decompress(payload)), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def quantize_symmetric(values: np.ndarray, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    flat = values.reshape(len(values), -1).astype(np.float32)
    limit = np.quantile(np.abs(flat), 0.999, axis=0).astype(np.float32)
    step = np.maximum(limit / float(maximum), 1e-8)
    dtype = np.int8 if maximum <= 127 else np.int16
    codes = np.rint(np.clip(flat / step, -maximum, maximum)).astype(dtype)
    return codes.reshape(values.shape), step


def dequantize_symmetric(codes: np.ndarray, step: np.ndarray) -> np.ndarray:
    flat = codes.reshape(len(codes), -1).astype(np.float32)
    return (flat * step.reshape(1, -1)).reshape(codes.shape)


def pack_int4(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = np.asarray(codes.shape, dtype=np.int32)
    values = (codes.reshape(-1).astype(np.int16) + 8).astype(np.uint8)
    if len(values) % 2:
        values = np.concatenate([values, np.asarray([8], dtype=np.uint8)])
    return values[0::2] | (values[1::2] << 4), shape


def unpack_int4(packed: np.ndarray, shape: np.ndarray) -> np.ndarray:
    values = np.empty(2 * len(packed), dtype=np.uint8)
    values[0::2] = packed & 15
    values[1::2] = packed >> 4
    count = int(np.prod(shape))
    return (values[:count].astype(np.int8) - 8).reshape(tuple(shape.tolist()))


def quantize_q4(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Match the evaluated stream: requantize the robust int8 representation.
    original, old_step = quantize_symmetric(values, 127)
    step = np.maximum(old_step * (127.0 / 7.0), 1e-8)
    flat = original.reshape(len(original), -1).astype(np.float32)
    codes = np.rint(np.clip(flat * old_step.reshape(1, -1) /
                           step.reshape(1, -1), -7, 7)).astype(np.int8).reshape(values.shape)
    packed, shape = pack_int4(codes)
    return packed, shape, step


def dequantize_q4(packed: np.ndarray, shape: np.ndarray, step: np.ndarray) -> np.ndarray:
    return dequantize_symmetric(unpack_int4(packed, shape), step)


def quantize_xyz(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    normalized = (values - low) / np.maximum(high - low, 1e-8)
    return np.rint(np.clip(normalized, 0.0, 1.0) * 65535.0).astype(np.uint16)


def dequantize_xyz(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return low + values.astype(np.float32) / 65535.0 * (high - low)


def quantize_quaternion(values: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype(np.int16)


def dequantize_quaternion(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.float32) / 32767.0
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-8)
    return result


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ], axis=-1)


def quaternion_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    value = quaternion / np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-8)
    value = np.where(value[:, :1] < 0.0, -value, value)
    vector_norm = np.linalg.norm(value[:, 1:], axis=1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(value[:, :1], 0.0, 1.0))
    axis = value[:, 1:] / np.maximum(vector_norm, 1e-8)
    return (axis * angle).astype(np.float32)


def rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec, axis=1, keepdims=True)
    half = 0.5 * angle
    scale = np.sin(half) / np.maximum(angle, 1e-8)
    quaternion = np.concatenate([np.cos(half), rotvec * scale], axis=1)
    quaternion[angle[:, 0] < 1e-8] = np.asarray([1.0, 0.0, 0.0, 0.0], np.float32)
    return quaternion.astype(np.float32)
