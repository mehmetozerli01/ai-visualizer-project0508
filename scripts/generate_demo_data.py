#!/usr/bin/env python3
"""
AI Visualizer — sentetik demo veri üretici.

Çalıştırma (proje kökünden)::

    python scripts/generate_demo_data.py

Çıktılar ``data/raw/`` altına yazılır:
  - blood_cell_features.csv  — kan hücresi / kümeleme demosu
  - generic_server_logs.csv  — zaman serisi anomali demosu
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def generate_blood_cell_features(*, n_rows: int = 240, seed: int = 42) -> pd.DataFrame:
    """Lökosit/eritrosit benzeri özellik vektörleri (HealthTech kümeleme demosu)."""
    rng = np.random.default_rng(seed)
    n_healthy = int(n_rows * 0.72)
    n_abnormal = n_rows - n_healthy

    def _block(count: int, *, abnormal: bool) -> pd.DataFrame:
        if abnormal:
            diam = rng.normal(16.5, 3.2, count).clip(6, 28)
            nucleus = rng.normal(0.78, 0.12, count).clip(0.35, 1.0)
            red = rng.normal(0.82, 0.14, count).clip(0.2, 1.0)
            irreg = rng.normal(0.62, 0.15, count).clip(0.25, 1.0)
        else:
            diam = rng.normal(8.2, 1.1, count).clip(5, 14)
            nucleus = rng.normal(0.42, 0.08, count).clip(0.15, 0.75)
            red = rng.normal(0.55, 0.1, count).clip(0.25, 0.85)
            irreg = rng.normal(0.18, 0.07, count).clip(0.02, 0.45)
        return pd.DataFrame(
            {
                "cell_id": np.arange(count, dtype=int),
                "cell_diameter_um": diam,
                "nucleus_density": nucleus,
                "color_intensity_red": red,
                "irregularity_score": irreg,
            }
        )

    healthy = _block(n_healthy, abnormal=False)
    abnormal = _block(n_abnormal, abnormal=True)
    healthy["cell_id"] = np.arange(len(healthy))
    abnormal["cell_id"] = np.arange(len(healthy), len(healthy) + len(abnormal))
    out = pd.concat([healthy, abnormal], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out["cell_id"] = np.arange(len(out))
    return out


def generate_generic_server_logs(*, n_rows: int = 720, seed: int = 43) -> pd.DataFrame:
    """Sunucu CPU/RAM metrikleri + zaman damgası (zaman serisi anomali demosu)."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-06-01", periods=n_rows, freq="5min")
    cpu = rng.normal(42.0, 11.0, n_rows).clip(2.0, 88.0)
    ram = rng.normal(58.0, 9.0, n_rows).clip(10.0, 92.0)
    disk = rng.lognormal(mean=2.1, sigma=0.45, size=n_rows).clip(0.5, 120.0)
    net = rng.lognormal(mean=2.8, sigma=0.35, size=n_rows).clip(1.0, 250.0)

    spike_idx = rng.choice(n_rows, size=max(12, n_rows // 60), replace=False)
    for i in spike_idx:
        cpu[i] = float(rng.uniform(93.0, 99.5))
        ram[i] = float(rng.uniform(91.0, 99.0))
        disk[i] = float(rng.uniform(95.0, 118.0))

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "cpu_percent": np.round(cpu, 2),
            "ram_percent": np.round(ram, 2),
            "disk_io_mbps": np.round(disk, 2),
            "network_mbps": np.round(net, 2),
        }
    )


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    blood_path = RAW_DIR / "blood_cell_features.csv"
    logs_path = RAW_DIR / "generic_server_logs.csv"

    blood_df = generate_blood_cell_features()
    logs_df = generate_generic_server_logs()

    blood_df.to_csv(blood_path, index=False)
    logs_df.to_csv(logs_path, index=False)

    print(f"Wrote {blood_path} ({len(blood_df)} rows)")
    print(f"Wrote {logs_path} ({len(logs_df)} rows)")


if __name__ == "__main__":
    main()
