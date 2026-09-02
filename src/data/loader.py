"""Memory-conscious loaders and file inspection for the IEEE-CIS dataset."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)

# The project intentionally uses a focused raw feature subset so a student/local machine does not
# need to materialize all 394 transaction columns. The full raw files remain available for EDA.
TRANSACTION_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "P_emaildomain",
    "R_emaildomain",
    "C1",
    "C2",
    "D1",
    "D2",
]
IDENTITY_COLUMNS = [
    "TransactionID",
    "DeviceType",
    "DeviceInfo",
    "id_01",
    "id_02",
    "id_31",
]

_TRANSACTION_DTYPES: dict[str, str] = {
    "TransactionID": "int64",
    "isFraud": "int8",
    "TransactionDT": "int32",
    "TransactionAmt": "float32",
    "ProductCD": "category",
    "card1": "int32",
    "card2": "float32",
    "card3": "float32",
    "card4": "category",
    "card5": "float32",
    "card6": "category",
    "addr1": "float32",
    "addr2": "float32",
    "dist1": "float32",
    "P_emaildomain": "category",
    "R_emaildomain": "category",
    "C1": "float32",
    "C2": "float32",
    "D1": "float32",
    "D2": "float32",
}
_IDENTITY_DTYPES: dict[str, str] = {
    "TransactionID": "int64",
    "DeviceType": "category",
    "DeviceInfo": "category",
    "id_01": "float32",
    "id_02": "float32",
    "id_31": "category",
}


def _existing_columns(path: Path, wanted: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns
    return [c for c in wanted if c in header]


def _normalise_identity_names(columns: list[str]) -> dict[str, str]:
    """Map Kaggle test identity names such as id-01 to the training id_01 convention."""
    mapping: dict[str, str] = {}
    for col in columns:
        if col.startswith("id-"):
            mapping[col] = "id_" + col[3:]
    return mapping


def optimize_dtypes(df: pd.DataFrame, *, copy: bool = False) -> pd.DataFrame:
    """Downcast numerics and categorise strings, avoiding a full copy by default."""
    out = df.copy() if copy else df
    for col in out.select_dtypes(include=["integer"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in out.select_dtypes(include=["floating"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="float")
    for col in out.select_dtypes(include=["object", "string"]).columns:
        nunique = out[col].nunique(dropna=True)
        if len(out) and nunique / len(out) < 0.5:
            out[col] = out[col].astype("category")
    return out


def _read_csv_selected(
    path: Path,
    columns: list[str],
    dtype_map: dict[str, str],
    nrows: int | None = None,
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """Read selected columns in bounded chunks to avoid pandas CSV-parser memory spikes."""
    available = _existing_columns(path, columns)
    dtypes = {name: dtype_map[name] for name in available if name in dtype_map}
    reader = pd.read_csv(
        path,
        usecols=available,
        dtype=dtypes,
        nrows=nrows,
        chunksize=chunk_size,
        low_memory=True,
    )
    parts = list(reader)
    if not parts:
        return pd.DataFrame(columns=available)
    if len(parts) == 1:
        return parts[0]
    # copy=False is advisory; it still avoids the much larger parser peak caused by one-shot reads.
    return pd.concat(parts, ignore_index=True, copy=False)


def load_ieee_cis(
    raw_dir: Path | None = None,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Load selected labeled IEEE-CIS columns and safely left-join identity features."""
    raw_dir = Path(raw_dir or settings.raw_dir)
    tx_path = raw_dir / "train_transaction.csv"
    id_path = raw_dir / "train_identity.csv"
    if not tx_path.exists():
        raise FileNotFoundError(
            f"Missing {tx_path}. Place the IEEE-CIS CSV files in data/raw/."
        )

    nrows = sample_size if sample_size and sample_size > 0 else None
    tx = _read_csv_selected(tx_path, TRANSACTION_COLUMNS, _TRANSACTION_DTYPES, nrows=nrows)
    if id_path.exists():
        identity = _read_csv_selected(id_path, IDENTITY_COLUMNS, _IDENTITY_DTYPES)
        # A sampled transaction frame should not carry unrelated identity rows through the merge.
        if nrows is not None:
            identity = identity[identity["TransactionID"].isin(tx["TransactionID"])]
        tx = tx.merge(identity, on="TransactionID", how="left", validate="one_to_one", copy=False)
        del identity
    return optimize_dtypes(tx, copy=False)


def load_ieee_official_test(
    raw_dir: Path | None = None,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Load the unlabeled official Kaggle test files using the training identity naming convention."""
    raw_dir = Path(raw_dir or settings.raw_dir)
    tx_path = raw_dir / "test_transaction.csv"
    id_path = raw_dir / "test_identity.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"Missing {tx_path}")

    test_tx_columns = [c for c in TRANSACTION_COLUMNS if c != "isFraud"]
    nrows = sample_size if sample_size and sample_size > 0 else None
    tx = _read_csv_selected(tx_path, test_tx_columns, _TRANSACTION_DTYPES, nrows=nrows)

    if id_path.exists():
        header = pd.read_csv(id_path, nrows=0).columns.tolist()
        rename_map = _normalise_identity_names(header)
        reverse = {new: old for old, new in rename_map.items()}
        wanted_source = [reverse.get(c, c) for c in IDENTITY_COLUMNS]
        available_source = [c for c in wanted_source if c in header]
        source_dtypes: dict[str, str] = {}
        for source in available_source:
            normalized = rename_map.get(source, source)
            if normalized in _IDENTITY_DTYPES:
                source_dtypes[source] = _IDENTITY_DTYPES[normalized]
        identity = pd.read_csv(
            id_path,
            usecols=available_source,
            dtype=source_dtypes,
            low_memory=False,
        ).rename(columns=rename_map)
        if nrows is not None:
            identity = identity[identity["TransactionID"].isin(tx["TransactionID"])]
        tx = tx.merge(identity, on="TransactionID", how="left", validate="one_to_one", copy=False)
        del identity
    return optimize_dtypes(tx, copy=False)


def _count_csv_rows(path: Path) -> int:
    """Count data rows without loading a CSV into memory."""
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return SHA-256 for reproducible dataset/file metadata."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_ieee_files(raw_dir: Path | None = None, *, hash_training_files: bool = False) -> dict[str, Any]:
    """Inspect actual IEEE-CIS filenames, row/column counts and train/test schema facts."""
    raw_dir = Path(raw_dir or settings.raw_dir)
    names = [
        "train_transaction.csv",
        "train_identity.csv",
        "test_transaction.csv",
        "test_identity.csv",
    ]
    files: dict[str, Any] = {}
    for name in names:
        path = raw_dir / name
        if not path.exists():
            files[name] = {"exists": False}
            continue
        header = pd.read_csv(path, nrows=0).columns.tolist()
        normalized = [(_normalise_identity_names([c]).get(c, c)) for c in header]
        item: dict[str, Any] = {
            "exists": True,
            "size_bytes": path.stat().st_size,
            "rows": _count_csv_rows(path),
            "columns": len(header),
            "has_transaction_id": "TransactionID" in header,
            "has_target": "isFraud" in header,
            "normalized_identity_schema": normalized if "identity" in name else None,
        }
        if hash_training_files and name in {"train_transaction.csv", "train_identity.csv"}:
            item["sha256"] = file_sha256(path)
        files[name] = item
    return files



def load_training_data(sample_size: int | None = None) -> pd.DataFrame:
    """Load the real labeled IEEE-CIS training data.

    The repository intentionally ships without raw CSVs. Place the official files in
    ``data/raw/`` (or set ``RAW_DATA_DIR``) before calling this function. ``sample_size``
    is an optional first-N-row debug aid that still uses the real IEEE-CIS CSV; the normal
    training command leaves it unset and trains on the full labeled dataset.
    """
    return load_ieee_cis(settings.raw_dir, sample_size=sample_size)
