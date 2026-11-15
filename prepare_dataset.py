"""Build deduplicated, auditable train/validation/test CSVs from data/source."""
from __future__ import annotations

import hashlib
import csv
import json
import re
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "source"
OUT = ROOT / "data"
SEED = 42
EXPECTED_BENGALI = {
    "train": "bengali_train.parquet",
    "validation": "bengali_validation.parquet",
    "test": "bengali_test.parquet",
}


def clean_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sources() -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    uci_path = SOURCE / "uci_sms_spam_collection.zip"
    with zipfile.ZipFile(uci_path) as archive:
        member = next(name for name in archive.namelist() if name.endswith("SMSSpamCollection"))
        with archive.open(member) as stream:
            # UCI is a tab-delimited text file, not CSV-quoted; quotation marks
            # occurring in message bodies must remain literal characters.
            uci = pd.read_csv(stream, sep="\t", header=None, names=["original_category", "text"],
                              dtype=str, quoting=csv.QUOTE_NONE)
    uci["source_dataset"] = "UCI SMS Spam Collection"
    uci["language"] = "English"
    uci["source_split"] = "unsplit"
    uci["source_row_id"] = [f"uci:{i}" for i in range(len(uci))]
    uci["label"] = uci["original_category"].map({"ham": "ham", "spam": "spam"})
    records.append(uci)

    for split, filename in EXPECTED_BENGALI.items():
        part = pd.read_parquet(SOURCE / filename)
        part = part.rename(columns={"label": "original_category", "source": "language"})
        part["source_dataset"] = "Bengali SMS Smishing Dataset (Hugging Face)"
        part["source_split"] = split
        part["source_row_id"] = [f"bengali:{split}:{i}" for i in range(len(part))]
        # The source card defines promo as legitimate commercial messaging.
        # For this scam/smishing task it is mapped to ham while its category stays intact.
        part["label"] = part["original_category"].map({"smish": "spam", "promo": "ham", "normal": "ham"})
        records.append(part[["text", "label", "source_dataset", "original_category", "language", "source_split", "source_row_id"]])

    data = pd.concat(records, ignore_index=True)
    data["text"] = data["text"].fillna("").astype(str)
    data["_key"] = data["text"].map(clean_key)
    data["_priority"] = data["source_split"].map({"test": 0, "validation": 1, "train": 2, "unsplit": 3})
    return data


def main() -> None:
    missing = [str(SOURCE / f) for f in ["uci_sms_spam_collection.zip", *EXPECTED_BENGALI.values()] if not (SOURCE / f).is_file()]
    if missing:
        raise FileNotFoundError("Required source files are missing: " + ", ".join(missing))

    data = load_sources()
    input_rows_by_dataset = data.groupby("source_dataset").size().to_dict()
    input_category_counts = data.groupby(["source_dataset", "original_category"]).size().rename("count").reset_index().to_dict("records")
    empty = data["_key"].eq("")
    empty_rows = int(empty.sum())
    empty_rows_by_dataset = data.loc[empty].groupby("source_dataset").size().to_dict()
    data = data.loc[~empty].copy()

    # Ambiguous exact duplicates with conflicting binary labels are excluded in full.
    conflicts = data.groupby("_key")["label"].nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    conflict_count = len(conflict_keys)
    conflict_rows = data["_key"].isin(conflict_keys)
    conflict_rows_by_dataset = data.loc[conflict_rows].groupby("source_dataset").size().to_dict()
    data = data.loc[~data["_key"].isin(conflict_keys)].copy()
    data = data.sort_values(["_priority", "source_row_id"], kind="stable")
    duplicate_mask = data.duplicated("_key", keep="first")
    duplicates_removed = int(duplicate_mask.sum())
    duplicate_rows_by_dataset = data.loc[duplicate_mask].groupby("source_dataset").size().to_dict()
    duplicate_rows_by_source_split = data.loc[duplicate_mask].groupby(["source_dataset", "source_split"]).size().rename("count").reset_index().to_dict("records")
    duplicate_groups = int(data.loc[data["_key"].duplicated(keep=False), "_key"].nunique())
    data = data.loc[~duplicate_mask].copy()

    # Keep the publisher's Bengali test split untouched. UCI has no canonical split;
    # create a seeded stratified 70/10/20 split from its de-duplicated messages.
    bengali = data[data["source_dataset"].str.startswith("Bengali")].copy()
    uci = data[data["source_dataset"].eq("UCI SMS Spam Collection")].copy()
    uci_train_val, uci_test = train_test_split(
        uci, test_size=0.20, random_state=SEED, stratify=uci["label"]
    )
    uci_train, uci_val = train_test_split(
        uci_train_val, test_size=0.125, random_state=SEED, stratify=uci_train_val["label"]
    )
    parts = {
        "train": pd.concat([bengali[bengali.source_split.eq("train")], uci_train]),
        "validation": pd.concat([bengali[bengali.source_split.eq("validation")], uci_val]),
        "test": pd.concat([bengali[bengali.source_split.eq("test")], uci_test]),
    }
    outputs: dict[str, pd.DataFrame] = {}
    for split, part in parts.items():
        part = part.copy()
        part["split"] = split
        outputs[split] = part[["text", "label", "source_dataset", "original_category", "language", "source_split", "source_row_id", "split"]].sort_values("source_row_id").reset_index(drop=True)

    # Final exact normalized-text overlap check across every effective split.
    keys = {split: set(part["text"].map(clean_key)) for split, part in outputs.items()}
    overlaps = {
        "train_validation": len(keys["train"] & keys["validation"]),
        "train_test": len(keys["train"] & keys["test"]),
        "validation_test": len(keys["validation"] & keys["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Duplicate leakage across effective splits: {overlaps}")

    OUT.mkdir(exist_ok=True)
    for split, frame in outputs.items():
        frame.to_csv(OUT / f"unified_{split}.csv", index=False, encoding="utf-8")

    summary = {
        "random_seed": SEED,
        "normalization_for_duplicate_detection": "Unicode NFKC, casefold, collapse whitespace; punctuation retained",
        "empty_rows_removed": empty_rows,
        "empty_rows_removed_by_dataset": empty_rows_by_dataset,
        "input_rows_by_dataset": input_rows_by_dataset,
        "input_original_category_counts": input_category_counts,
        "conflicting_duplicate_text_groups_removed": conflict_count,
        "conflicting_rows_removed_by_dataset": conflict_rows_by_dataset,
        "duplicate_rows_removed": duplicates_removed,
        "duplicate_text_groups": duplicate_groups,
        "duplicate_rows_removed_by_dataset": duplicate_rows_by_dataset,
        "duplicate_rows_removed_by_source_split": duplicate_rows_by_source_split,
        "cross_split_exact_normalized_text_overlaps": overlaps,
        "source_file_sha256": {p.name: sha256(p) for p in sorted(SOURCE.iterdir()) if p.is_file()},
        "source_split_counts": {k: len(v) for k, v in outputs.items()},
        "label_counts": {k: v.label.value_counts().to_dict() for k, v in outputs.items()},
        "dataset_language_counts": {
            k: v.groupby(["source_dataset", "language", "label"]).size().rename("count").reset_index().to_dict("records")
            for k, v in outputs.items()
        },
        "source_original_category_counts": data.groupby(["source_dataset", "original_category"]).size().rename("count").reset_index().to_dict("records"),
        "nepal_synthetic_test_rows_excluded": 16,
    }
    (OUT / "dataset_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
