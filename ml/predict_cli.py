"""
Prediction CLI — for server.ts /api/recommend integration
===========================================================
Accepts context variables as a JSON string in ``argv[1]``, runs the full
``predict_and_recommend`` pipeline, and prints the result as JSON to stdout.

Usage (called by server.ts):
    python3 -m ml.predict_cli '<context_json>' [artifact_dir]
"""

from __future__ import annotations

import json
import logging
import sys
import traceback

import numpy as np

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python3 -m ml.predict_cli '<context_json>' [artifact_dir]"}))
        sys.exit(1)

    try:
        context: dict = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON context: {exc}"}))
        sys.exit(1)

    artifact_dir = sys.argv[2] if len(sys.argv) > 2 else "ml/artifacts"

    try:
        from ml.predict_and_recommend import predict_and_recommend

        result = predict_and_recommend(context, artifact_dir=artifact_dir)
        print(json.dumps(result.to_dict(), default=_json_safe))
    except Exception as exc:
        print(json.dumps({
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }, default=_json_safe))
        sys.exit(1)


if __name__ == "__main__":
    main()
