"""
data/loader.py - SECOM Dataset Downloader & Loader
====================================================

Downloads the SECOM semiconductor manufacturing dataset from the UCI
Machine Learning Repository, parses it, and provides a clean API for
the rest of the pipeline.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


class SECOMDataLoader:
    """Download, parse, and serve the SECOM dataset.

    Attributes:
        config: Global configuration object.
        data_path: Path to the raw secom.data file.
        labels_path: Path to the raw secom_labels.data file.
    """

    def __init__(self, cfg=None):
        """Initialise paths from configuration.

        Args:
            cfg: Optional alternate config; defaults to the global singleton.
        """
        self.config = cfg or config
        self.data_path = os.path.join(self.config.raw_data_dir, "secom.data")
        self.labels_path = os.path.join(self.config.raw_data_dir, "secom_labels.data")

    # ─────────────────────────────────────────────────────────────────
    # Downloading
    # ─────────────────────────────────────────────────────────────────
    def download_data(self) -> None:
        """Download both SECOM files from UCI with retry logic.

        Creates the raw-data directory if needed.  Skips files that
        already exist on disk.  Uses ``tqdm`` for a progress bar and
        retries up to 3 times on network failure.

        Raises:
            RuntimeError: If a file cannot be downloaded after 3 attempts.
        """
        os.makedirs(self.config.raw_data_dir, exist_ok=True)

        files = [
            (self.config.secom_data_url, self.data_path, "secom.data"),
            (self.config.secom_labels_url, self.labels_path, "secom_labels.data"),
        ]

        for url, dest, name in files:
            if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                print(f"  ✓ {name} already exists – skipping download.")
                continue

            success = False
            for attempt in range(1, 4):
                try:
                    print(f"  ↓ Downloading {name} (attempt {attempt}/3) …")
                    resp = requests.get(url, stream=True, timeout=60)
                    resp.raise_for_status()

                    total = int(resp.headers.get("content-length", 0))
                    with open(dest, "wb") as fh, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=name,
                        leave=True,
                    ) as bar:
                        for chunk in resp.iter_content(chunk_size=8192):
                            fh.write(chunk)
                            bar.update(len(chunk))

                    if os.path.getsize(dest) == 0:
                        raise RuntimeError(f"Downloaded file {name} is empty.")

                    print(f"  ✓ {name} saved ({os.path.getsize(dest):,} bytes)")
                    success = True
                    break

                except (requests.RequestException, RuntimeError) as exc:
                    print(f"  ✗ Attempt {attempt} failed: {exc}")
                    if attempt < 3:
                        time.sleep(2 ** attempt)

            if not success:
                raise RuntimeError(
                    f"Failed to download {name} after 3 attempts. "
                    "Please download manually from the UCI repository."
                )

    # ─────────────────────────────────────────────────────────────────
    # Loading & parsing
    # ─────────────────────────────────────────────────────────────────
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Load raw SECOM data and labels into DataFrames.

        Returns:
            A tuple of ``(X, y, df_master)`` where:

            - **X** – sensor readings, columns ``F001`` … ``F591``.
            - **y** – binary target (0 = Pass, 1 = Fail).
            - **df_master** – X + y + timestamp, sorted by timestamp.
        """
        print("\n─── Loading SECOM data ───")

        # --- sensor data ------------------------------------------------
        X = pd.read_csv(
            self.data_path,
            sep=r"\s+",
            header=None,
            na_values=["NaN", "nan"],
        )
        feature_names = [f"F{i:03d}" for i in range(1, X.shape[1] + 1)]
        X.columns = feature_names
        print(f"  Sensor matrix shape : {X.shape}")

        # --- labels & timestamps ----------------------------------------
        labels_df = pd.read_csv(
            self.labels_path,
            sep=r"\s+",
            header=None,
            names=["label", "timestamp"],
        )

        # Parse timestamps – try multiple formats
        try:
            labels_df["timestamp"] = pd.to_datetime(
                labels_df["timestamp"], format="%d/%m/%Y %H:%M:%S"
            )
        except ValueError:
            labels_df["timestamp"] = pd.to_datetime(
                labels_df["timestamp"], format="mixed", dayfirst=True
            )

        # Recode: -1→0 (Pass), +1→1 (Fail)
        mapped = labels_df["label"].map({-1: 0, 1: 1})
        n_unmapped = int(mapped.isna().sum())
        if n_unmapped > 0:
            bad_vals = labels_df.loc[mapped.isna(), "label"].unique().tolist()
            raise ValueError(
                f"{n_unmapped} label(s) could not be mapped. "
                f"Expected -1 or 1, but found: {bad_vals}"
            )
        labels_df["label"] = mapped.astype(int)

        y = labels_df["label"]
        timestamps = labels_df["timestamp"]

        # --- master DataFrame -------------------------------------------
        df_master = X.copy()
        df_master["label"] = y.values
        df_master["timestamp"] = timestamps.values

        # Temporal sort
        df_master.sort_values("timestamp", inplace=True)
        df_master.reset_index(drop=True, inplace=True)

        y = df_master["label"]
        X = df_master[feature_names].copy()
        timestamps = df_master["timestamp"]

        # --- summary statistics -----------------------------------------
        n_pass = int((y == 0).sum())
        n_fail = int((y == 1).sum())
        total_missing = int(X.isna().sum().sum())
        total_cells = int(X.shape[0] * X.shape[1])
        features_gt50 = int((X.isna().mean() > 0.50).sum())

        print(f"  Date range          : {timestamps.min()} → {timestamps.max()}")
        print(f"  Duration            : {(timestamps.max() - timestamps.min()).days} days")
        print(f"  Pass (0)            : {n_pass:,} ({100 * n_pass / len(y):.1f}%)")
        print(f"  Fail (1)            : {n_fail:,} ({100 * n_fail / len(y):.1f}%)")
        print(f"  Imbalance ratio     : 1 : {n_pass / max(n_fail, 1):.1f}")
        print(f"  Missing values      : {total_missing:,} / {total_cells:,} ({100 * total_missing / total_cells:.1f}%)")
        print(f"  Features >50% miss  : {features_gt50}")

        return X, y, df_master

    # ─────────────────────────────────────────────────────────────────
    # Feature groups
    # ─────────────────────────────────────────────────────────────────
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return approximate sensor groups based on index ranges.

        Returns:
            Dictionary mapping group name → list of ``Fxxx`` strings.
        """
        groups: Dict[str, List[str]] = {
            "sensors_batch_1": [f"F{i:03d}" for i in range(1, 101)],
            "sensors_batch_2": [f"F{i:03d}" for i in range(101, 201)],
            "sensors_batch_3": [f"F{i:03d}" for i in range(201, 301)],
            "sensors_batch_4": [f"F{i:03d}" for i in range(301, 401)],
            "sensors_batch_5": [f"F{i:03d}" for i in range(401, 501)],
            "sensors_batch_6": [f"F{i:03d}" for i in range(501, 592)],
        }
        return groups

    # ─────────────────────────────────────────────────────────────────
    # Temporal features
    # ─────────────────────────────────────────────────────────────────
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract calendar and shift features from the timestamp column.

        Args:
            df: DataFrame that must contain a ``timestamp`` column.

        Returns:
            Same DataFrame with new time-feature columns appended.
        """
        ts = pd.to_datetime(df["timestamp"])

        df = df.copy()
        df["hour_of_day"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["shift"] = pd.cut(
            ts.dt.hour, bins=[-1, 8, 16, 24], labels=[0, 1, 2]
        ).cat.codes
        df["week_number"] = ts.dt.isocalendar().week.astype(int)
        df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        df["production_sequence"] = np.arange(len(df))

        print(f"  ✓ Time features added: hour_of_day, day_of_week, shift, "
              "week_number, is_weekend, production_sequence")
        return df

    # ─────────────────────────────────────────────────────────────────
    # Persistence helpers
    # ─────────────────────────────────────────────────────────────────
    def save_processed(
        self, X: pd.DataFrame, y: pd.Series, filepath: str
    ) -> None:
        """Save processed data to Parquet for fast reloading.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            filepath: Base path (``_X.parquet`` / ``_y.parquet`` appended).
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        X.to_parquet(filepath + "_X.parquet", engine="pyarrow")
        y.to_frame("label").to_parquet(filepath + "_y.parquet", engine="pyarrow")
        print(f"  ✓ Processed data saved → {filepath}_*.parquet")

    def load_processed(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load previously saved Parquet files.

        Args:
            filepath: Base path used in :meth:`save_processed`.

        Returns:
            ``(X, y)`` tuple.
        """
        X = pd.read_parquet(filepath + "_X.parquet", engine="pyarrow")
        y = pd.read_parquet(filepath + "_y.parquet", engine="pyarrow")["label"]
        print(f"  ✓ Processed data loaded from {filepath}_*.parquet")
        return X, y
