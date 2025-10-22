"""Quick diagnostic runner that invokes the notebook analyzer to produce
`data/results/bedrock_client_diag.json`. Useful for checking Bedrock API
key or boto3 client shape in the running environment.

Usage: from the repo root run:

    python scripts/diag_bedrock.py

It will load config/config.yaml, set BEDROCK_API_KEY from env if present,
then run analysis on a small in-memory notebook sample.
"""

import os
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` imports work when running
# with system python or the venv python.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.base import config_manager
from src.analyzers.notebook_analyzer import analyze_notebooks_batch


def main():
    # Ensure sample output dir exists
    os.makedirs("data/results", exist_ok=True)

    cfg = config_manager.config

    # Example: create a fake notebook entry
    sample = [
        {
            "name": "sample_notebook",
            "path": "notebooks/sample_notebook.py",
            "content": """
# dbutils demonstration
%sql
SELECT * FROM table

print('hello world')
""",
            "language": "python",
        }
    ]

    results = analyze_notebooks_batch(sample, cfg)

    out_path = "data/results/notebook_analysis_sample.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote sample analysis to {out_path}")


if __name__ == "__main__":
    main()
