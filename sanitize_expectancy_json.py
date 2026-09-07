#!/usr/bin/env python3
import json
import math
from pathlib import Path

P = Path("research/expectancy_filter_summary.json")


def clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        if math.isinf(x):
            return "+INF" if x > 0 else "-INF"
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x

obj = json.loads(P.read_text(encoding="utf-8"))
P.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
print("Sanitized expectancy_filter_summary.json to strict JSON")
