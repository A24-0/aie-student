from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


class ExperimentTracker:

    def __init__(self, artifacts_dir: Path) -> None:
        self.dir = Path(artifacts_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / "experiments.csv"
        self.jsonl_path = self.dir / "experiments.jsonl"

    def log_run(self, name: str, params: dict, metrics: dict) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row = {"run": name, "timestamp": ts, **params, **metrics}

        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"params": params, "metrics": metrics, "run": name, "timestamp": ts}, ensure_ascii=False) + "\n")

        write_header = not self.csv_path.exists()
        existing = []
        if not write_header:
            with self.csv_path.open(encoding="utf-8") as fh:
                existing = list(csv.DictReader(fh))
        fieldnames = list(row.keys())
        for r in existing:
            for key in r:
                if key not in fieldnames:
                    fieldnames.append(key)
        with self.csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in existing:
                writer.writerow(r)
            writer.writerow(row)

    def reset(self) -> None:
        for p in (self.csv_path, self.jsonl_path):
            if p.exists():
                p.unlink()
