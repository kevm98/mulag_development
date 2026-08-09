#!/usr/bin/env python3
"""Replace selected trajectory shapes with validated shapes from another library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("target", type=Path)
p.add_argument("source", type=Path)
p.add_argument("--shapes", required=True)
a = p.parse_args()

target = json.loads(a.target.read_text())
source = json.loads(a.source.read_text())
for name in (part.strip() for part in a.shapes.split(",") if part.strip()):
    target["shapes"][name] = source["shapes"][name]
    target["plant_reachability_report"][name] = source["plant_reachability_report"][name]
target.setdefault("generation_contract", {})["preserved_shapes"] = {
    "source": str(a.source.resolve()),
    "names": [part.strip() for part in a.shapes.split(",") if part.strip()],
}
a.target.write_text(json.dumps(target, indent=2) + "\n")
