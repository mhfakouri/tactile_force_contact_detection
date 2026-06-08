"""
Tactile/Force Signal Filtering and Contact Event Detection

This script generates synthetic force/torque-like sensor data, filters the noisy
force signal, detects contact onset/release, extracts simple contact features,
and saves publication-ready plots and CSV outputs.

Run from the project root:
    python src/contact_detection.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    plots: Path
    results: Path


@dataclass(frozen=True)
class DetectionResult:
    onset_time: float
    release_time: float
    threshold: float
    baseline_mean: float
    baseline_std: float
    onset_index: int
    release_index: int


def get_paths() -> ProjectPaths:
    """Return project paths relative to this file."""
    root = Path(__file__).resolve().parents[1]
    return ProjectPaths(
        root=root,
        data=root / "data",
        plots=root / "plots",
        results=root / "results",
    )


def generate_synthetic_force_torque(
    fs: int = 200,
    duration: float = 10.0,
    contact_start: float = 3.20,
    contact_end: float = 7.10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate force/torque-like data for a simple tactile contact experiment.

    The z-force contains a smooth contact event with noise and low-frequency drift.
    x/y force and torque channels are correlated with the z-force to resemble a
    simple force/torque sensor signal during contact.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(0, duration, 1 / fs)

    ramp_duration = 0.25
    contact_profile = np.zeros_like(time)

    ramp_up = (time >= contact_start) & (time < contact_start + ramp_duration)
    plateau = (time >= contact_start + ramp_duration) & (time <= contact_end - ramp_duration)
    ramp_down = (time > contact_end - ramp_duration) & (time <= contact_end)

    contact_profile[ramp_up] = 0.5 * (1 - np.cos(np.pi * (time[ramp_up] - contact_start) / ramp_duration))
    contact_profile[plateau] = 1.0
    contact_profile[ramp_down] = 0.5 * (1 + np.cos(np.pi * (time[ramp_down] - (contact_end - ramp_duration)) / ramp_duration))

    slow_variation = 1.0 + 0.10 * np.sin(2 * np.pi * 0.45 * time)
    fz_clean = 8.5 * contact_profile * slow_variation

    drift = 0.15 * np.sin(2 * np.pi * 0.12 * time)
    fz_raw = fz_clean + drift + rng.normal(0, 0.35, size=time.size)
    fx_raw = 0.10 * fz_clean + rng.normal(0, 0.18, size=time.size)
    fy_raw = -0.07 * fz_clean + rng.normal(0, 0.16, size=time.size)

    tx_raw = 0.025 * fy_raw + rng.normal(0, 0.010, size=time.size)
    ty_raw = -0.030 * fx_raw + rng.normal(0, 0.010, size=time.size)
    tz_raw = 0.010 * fz_clean * np.sin(2 * np.pi * 0.70 * time) + rng.normal(0, 0.008, size=time.size)

    return pd.DataFrame(
        {
            "time_s": time,
            "Fx_N": fx_raw,
            "Fy_N": fy_raw,
            "Fz_N": fz_raw,
            "Tx_Nm": tx_raw,
            "Ty_Nm": ty_raw,
            "Tz_Nm": tz_raw,
            "contact_ground_truth": ((time >= contact_start) & (time <= contact_end)).astype(int),
        }
    )


def moving_average(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """Apply centered moving-average filtering."""
    if window_samples < 1:
        raise ValueError("window_samples must be at least 1")
    kernel = np.ones(window_samples) / window_samples
    return np.convolve(signal, kernel, mode="same")


def butterworth_lowpass(signal: np.ndarray, fs: int, cutoff_hz: float = 8.0, order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth low-pass filtering."""
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normalized_cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal)


def _first_stable_crossing(mask: np.ndarray, min_samples: int, start_index: int = 0) -> int | None:
    """Find first index where a Boolean mask stays true for min_samples."""
    for idx in range(start_index, len(mask) - min_samples + 1):
        if mask[idx : idx + min_samples].all():
            return idx
    return None


def detect_contact_event(
    time: np.ndarray,
    force_filtered: np.ndarray,
    fs: int,
    baseline_duration: float = 1.5,
    threshold_std_multiplier: float = 5.0,
    min_contact_duration: float = 0.06,
) -> DetectionResult:
    """Detect contact onset and release using a baseline-adaptive threshold."""
    baseline_samples = int(baseline_duration * fs)
    min_samples = max(1, int(min_contact_duration * fs))

    baseline = force_filtered[:baseline_samples]
    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline, ddof=1))
    threshold = baseline_mean + threshold_std_multiplier * baseline_std

    above_threshold = force_filtered > threshold
    onset_index = _first_stable_crossing(above_threshold, min_samples)
    if onset_index is None:
        raise RuntimeError("No stable contact onset detected. Try reducing the threshold multiplier.")

    below_threshold_after_onset = ~above_threshold
    release_index = _first_stable_crossing(
        below_threshold_after_onset,
        min_samples,
        start_index=onset_index + min_samples,
    )
    if release_index is None:
        release_index = len(time) - 1

    return DetectionResult(
        onset_time=float(time[onset_index]),
        release_time=float(time[release_index]),
        threshold=float(threshold),
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        onset_index=onset_index,
        release_index=release_index,
    )


def _rise_time_10_90(time: np.ndarray, force: np.ndarray, onset_idx: int, peak_idx: int) -> float:
    """Estimate force rise time from 10% to 90% of contact peak."""
    if peak_idx <= onset_idx:
        return float("nan")

    segment = force[onset_idx : peak_idx + 1]
    segment_time = time[onset_idx : peak_idx + 1]
    peak = np.max(segment)
    low_level = 0.10 * peak
    high_level = 0.90 * peak

    low_candidates = np.where(segment >= low_level)[0]
    high_candidates = np.where(segment >= high_level)[0]
    if low_candidates.size == 0 or high_candidates.size == 0:
        return float("nan")

    return float(segment_time[high_candidates[0]] - segment_time[low_candidates[0]])


def extract_features(df: pd.DataFrame, detection: DetectionResult, fs: int) -> pd.DataFrame:
    """Extract compact features from the detected contact interval."""
    time = df["time_s"].to_numpy()
    fz_raw = df["Fz_N"].to_numpy()
    fz_filtered = df["Fz_butterworth_N"].to_numpy()

    start = detection.onset_index
    stop = detection.release_index
    contact_time = time[start:stop]
    contact_force = fz_filtered[start:stop]

    if contact_force.size == 0:
        raise RuntimeError("Detected contact interval is empty.")

    peak_local_idx = int(np.argmax(contact_force))
    peak_idx = start + peak_local_idx
    impulse = float(np.trapezoid(contact_force, contact_time)) if contact_time.size > 1 else 0.0

    features = {
        "detected_onset_s": detection.onset_time,
        "detected_release_s": detection.release_time,
        "contact_duration_s": detection.release_time - detection.onset_time,
        "peak_force_raw_N": float(np.max(fz_raw[start:stop])),
        "peak_force_filtered_N": float(np.max(contact_force)),
        "mean_contact_force_N": float(np.mean(contact_force)),
        "force_impulse_Ns": impulse,
        "rise_time_10_90_s": _rise_time_10_90(time, fz_filtered, start, peak_idx),
        "baseline_mean_N": detection.baseline_mean,
        "baseline_std_N": detection.baseline_std,
        "detection_threshold_N": detection.threshold,
        "sampling_frequency_Hz": fs,
    }
    return pd.DataFrame([features])


def save_plots(df: pd.DataFrame, detection: DetectionResult, paths: ProjectPaths) -> None:
    """Save plots for filtering and contact-event detection."""
    time = df["time_s"]

    plt.figure(figsize=(10, 5))
    plt.plot(time, df["Fz_N"], label="Noisy raw Fz")
    plt.plot(time, df["Fz_moving_average_N"], label="Moving average")
    plt.plot(time, df["Fz_butterworth_N"], label="Butterworth low-pass")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title("Force Signal Filtering")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.plots / "01_force_signal_filtering.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time, df["Fz_butterworth_N"], label="Filtered Fz")
    plt.axhline(detection.threshold, linestyle="--", label="Detection threshold")
    plt.axvline(detection.onset_time, linestyle=":", label=f"Onset: {detection.onset_time:.2f} s")
    plt.axvline(detection.release_time, linestyle=":", label=f"Release: {detection.release_time:.2f} s")
    plt.axvspan(detection.onset_time, detection.release_time, alpha=0.15, label="Detected contact")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title("Contact Event Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.plots / "02_contact_event_detection.png", dpi=180)
    plt.close()

    channels = ["Fx_N", "Fy_N", "Fz_N", "Tx_Nm", "Ty_Nm", "Tz_Nm"]
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 8), sharex=True)
    for axis, channel in zip(axes, channels):
        axis.plot(time, df[channel], label=channel)
        axis.axvspan(detection.onset_time, detection.release_time, alpha=0.12)
        axis.set_ylabel(channel)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Synthetic Force/Torque Sensor Channels")
    fig.tight_layout()
    fig.savefig(paths.plots / "03_force_torque_channels.png", dpi=180)
    plt.close(fig)


def run_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, DetectionResult]:
    """Run the full project pipeline and save outputs."""
    fs = 200
    paths = get_paths()
    paths.data.mkdir(exist_ok=True)
    paths.plots.mkdir(exist_ok=True)
    paths.results.mkdir(exist_ok=True)

    df = generate_synthetic_force_torque(fs=fs)
    df["Fz_moving_average_N"] = moving_average(df["Fz_N"].to_numpy(), window_samples=int(0.10 * fs))
    df["Fz_butterworth_N"] = butterworth_lowpass(df["Fz_N"].to_numpy(), fs=fs, cutoff_hz=8.0, order=4)

    detection = detect_contact_event(
        time=df["time_s"].to_numpy(),
        force_filtered=df["Fz_butterworth_N"].to_numpy(),
        fs=fs,
    )
    features = extract_features(df, detection, fs=fs)

    df.to_csv(paths.data / "synthetic_force_torque_data.csv", index=False)
    features.to_csv(paths.results / "contact_features_summary.csv", index=False)
    save_plots(df, detection, paths)

    print("Pipeline completed successfully.")
    print(f"Detected contact onset:  {detection.onset_time:.3f} s")
    print(f"Detected contact release: {detection.release_time:.3f} s")
    print(f"Saved data to:            {paths.data / 'synthetic_force_torque_data.csv'}")
    print(f"Saved features to:        {paths.results / 'contact_features_summary.csv'}")
    print(f"Saved plots to:           {paths.plots}")
    return df, features, detection


if __name__ == "__main__":
    run_pipeline()
