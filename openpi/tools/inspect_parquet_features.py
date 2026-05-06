#!/usr/bin/env python3
"""Inspect HuggingFace features (keys and types) stored in a Parquet file's schema metadata.

Usage:
    python tools/inspect_parquet_features.py <path_to.parquet>
    python tools/inspect_parquet_features.py  # uses default path below
"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def _format_feature(features_dict: dict, indent: int = 0) -> list[str]:
    """Format each feature as 'key: _type ...' and recurse into nested 'feature'."""
    lines = []
    prefix = "  " * indent
    for key, val in features_dict.items():
        if not isinstance(val, dict):
            lines.append(f"{prefix}{key}: (raw) {val}")
            continue
        _type = val.get("_type", "?")
        extra = []
        if "dtype" in val:
            extra.append(f"dtype={val['dtype']}")
        if "feature" in val:
            lines.append(f"{prefix}{key}: _type={_type}" + (f" ({', '.join(extra)})" if extra else ""))
            lines.extend(_format_feature(val["feature"], indent + 1))
        else:
            line = f"{prefix}{key}: _type={_type}"
            if extra:
                line += f" ({', '.join(extra)})"
            lines.append(line)
    return lines


def inspect_parquet_features(parquet_path: Path) -> None:
    """Print Arrow schema and HuggingFace features (keys + types) from a Parquet file."""
    if not parquet_path.exists():
        print(f"File not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    table = pq.read_table(parquet_path)
    schema = table.schema
    metadata = schema.metadata or {}

    print(f"File: {parquet_path}")
    print("=" * 60)

    # 1) Arrow schema (column names + pyarrow types)
    print("\n[Arrow schema]")
    for name, typ in zip(schema.names, schema.types):
        print(f"  {name}: {typ}")

    # 2) HuggingFace metadata
    if b"huggingface" not in metadata:
        print("\n[HuggingFace metadata]")
        print("  (none — no 'huggingface' key in schema metadata)")
        return

    try:
        hf_meta = json.loads(metadata[b"huggingface"].decode("utf-8"))
    except Exception as e:
        print(f"\n[HuggingFace metadata] Failed to decode JSON: {e}")
        return

    print("\n[HuggingFace metadata]")
    if "info" not in hf_meta:
        print("  (no 'info' in metadata)")
        return

    info = hf_meta["info"]
    features = info.get("features")
    if not features:
        print("  (no 'info.features' in metadata)")
        return

    # 3) Feature keys and types (and flag List)
    print("\n[Features: key → _type]")
    list_keys = []
    for key, feat in features.items():
        if isinstance(feat, dict) and "_type" in feat:
            t = feat["_type"]
            if t == "List":
                list_keys.append(key)
            inner = ""
            if "feature" in feat and isinstance(feat["feature"], dict):
                inner = feat["feature"].get("_type", "?")
                if "dtype" in feat["feature"]:
                    inner += f" ({feat['feature']['dtype']})"
            if inner:
                print(f"  {key}: _type={t}  →  feature: {inner}")
            else:
                print(f"  {key}: _type={t}")

    # 4) Formatted tree
    print("\n[Features tree]")
    for line in _format_feature(features):
        print(line)

    # 5) Warning if any List (incompatible with datasets.Features)
    if list_keys:
        print("\n⚠️  Incompatible with HuggingFace datasets:")
        print("    These keys use _type='List'; datasets only accepts 'Sequence' or 'LargeList':")
        for k in list_keys:
            print(f"      - {k}")
        print("    (Use the patch in data_loader.py or fix the parquet metadata to use 'Sequence'.)")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect HuggingFace features (keys and types) in a Parquet file's schema metadata."
    )
    parser.add_argument(
        "parquet_path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to a .parquet file (e.g. data/chunk-000/episode_000000.parquet)",
    )
    args = parser.parse_args()

    if args.parquet_path is None:
        # Example default; user can edit or pass path
        args.parquet_path = Path("data/chunk-000/episode_000000.parquet")
        if not args.parquet_path.exists():
            print("Usage: python tools/inspect_parquet_features.py <path_to.parquet>", file=sys.stderr)
            print("Example: python tools/inspect_parquet_features.py /path/to/repo/data/chunk-000/episode_000000.parquet", file=sys.stderr)
            sys.exit(1)

    inspect_parquet_features(args.parquet_path)


if __name__ == "__main__":
    main()
