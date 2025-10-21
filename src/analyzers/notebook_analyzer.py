"""Notebook analysis utilities.

Provides heuristic analysis for Databricks notebooks to identify
Databricks-specific syntax, estimate complexity, flag issues, and
suggest AWS equivalent approaches. Optionally calls Amazon Bedrock
when enabled in config (features.enable_bedrock_analysis). Bedrock
enrichment is best-effort and failures are returned in the
`bedrock_enrichment` field without failing the analysis pipeline.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from utils.base import ConfigManager

# Expanded list of common Databricks-specific constructs to detect
DB_SPECIFIC_PATTERNS = {
    "dbutils": r"\bdbutils\.",
    "display": r"\bdisplay\(",
    "displayHTML": r"\bdisplayHTML\(",
    "dbutils_widgets": r"\bdbutils\.widgets\.",
    "spark_sql_call": r"\bspark\.sql\(",
    "sql_cell_magic": r"^%sql",
    "mlflow": r"\bmlflow\.",
}


def _call_bedrock_model(
    prompt: str, config: ConfigManager, model_id: Optional[str] = None
) -> Dict[str, Any]:
    """Call AWS Bedrock model if configured.

    Returns a dictionary with either {'ok': True, 'model_id': .., 'result': parsed}
    or {'error': '...'} when invocation fails. Parsing attempts to decode JSON
    responses into Python objects when the model returns JSON text.
    """
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, PartialCredentialsError
    except Exception as e:
        return {"error": f"boto3 not available: {e}", "raw": None}

    # allow model_id override
    bedrock_cfg = (
        config.get("aws", {}).get("bedrock", {}) if hasattr(config, "get") else {}
    )
    configured_model = model_id or (
        bedrock_cfg.get("model_id") if isinstance(bedrock_cfg, dict) else None
    )
    if not configured_model:
        return {
            "error": "Bedrock model_id not configured in aws.bedrock.model_id",
            "raw": None,
        }

    # determine region from config or env
    region = (
        (config.get("aws", {}).get("region") if hasattr(config, "get") else None)
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
    )

    try:
        # Use a session so users can specify profile via AWS_PROFILE if desired
        session = boto3.Session()
        client = (
            session.client("bedrock", region_name=region)
            if region
            else session.client("bedrock")
        )
    except (NoCredentialsError, PartialCredentialsError) as e:
        return {"error": f"No AWS credentials available: {e}", "raw": None}
    except Exception as e:
        return {"error": f"failed to construct Bedrock client: {e}", "raw": None}

    try:
        # The Bedrock SDK can vary; prefer invoke_model/modelId/body or invoke depending on SDK
        response = client.invoke_model(
            modelId=configured_model, body=prompt.encode("utf-8")
        )
    except TypeError:
        # fallback naming
        try:
            response = client.invoke_model(
                ModelId=configured_model, Body=prompt.encode("utf-8")
            )
        except Exception as e:
            return {"error": f"bedrock call failed: {e}", "raw": None}
    except Exception as e:
        return {"error": f"bedrock call failed: {e}", "raw": None}

    # parse response body
    try:
        body = response.get("body")
        if hasattr(body, "read"):
            raw = body.read()
        else:
            raw = body

        # decode bytes
        try:
            decoded = (
                raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            )
        except Exception:
            decoded = str(raw)

        decoded = decoded.strip()
        if decoded.startswith("{") or decoded.startswith("["):
            try:
                parsed = json.loads(decoded)
            except Exception:
                parsed = {"text": decoded}
        else:
            parsed = {"text": decoded}

    except Exception as e:
        parsed = {"error": f"unable to parse bedrock body: {e}", "raw": str(response)}

    return {"ok": True, "model_id": configured_model, "result": parsed}


def analyze_notebook(
    content: str, path: str = "", language: str = "python"
) -> Dict[str, Any]:
    """Basic heuristic analysis for Databricks notebooks.

    Returns a dict with:
    - complexity: simple/medium/complex
    - issues: list of strings
    - suggestions: list of strings
    - detected_patterns: list of matched patterns
    - risk: green/yellow/red
    """
    issues: List[str] = []
    suggestions: List[str] = []
    detected: List[str] = []

    # Detect patterns
    for name, pat in DB_SPECIFIC_PATTERNS.items():
        try:
            if re.search(pat, content, flags=re.MULTILINE):
                detected.append(name)
        except re.error:
            # ignore regex problems
            continue

    # Heuristic complexity based on lines of code and keyword density
    loc = len(content.splitlines())
    complexity = "simple"
    if loc < 100:
        complexity = "simple"
    elif loc < 500:
        complexity = "medium"
    else:
        complexity = "complex"

    # Detect DB-specific constructs and flag issues/suggestions
    if "dbutils" in detected or re.search(r"\bdbutils\.", content):
        issues.append(
            "Uses dbutils - these utilities are Databricks-specific. Consider replacing with AWS native helpers (boto3 for S3, AWS Glue catalog APIs, or wrapper utilities)."
        )
        suggestions.append(
            "Replace dbutils calls with boto3 / custom wrappers or use Glue/Athena equivalents for data access."
        )

    if "display" in detected or re.search(r"\bdisplay\(", content):
        suggestions.append(
            "Uses display() for notebook visualization - on AWS consider rendering with matplotlib/plotly in notebooks or exporting results to dashboards (QuickSight) or building visualizations in a reporting layer."
        )

    if re.search(r"^%sql", content, flags=re.MULTILINE) or "sql_cell_magic" in detected:
        suggestions.append(
            "SQL cells detected - map to Glue/Athena/Redshift SQL depending on the target architecture; convert cell magics into programmatic SQL statements executed via boto3/pyathena or Spark on EMR."
        )

    if re.search(r"\bmlflow\.", content):
        suggestions.append(
            "Uses MLflow - on AWS you can use SageMaker Model Registry or integrate MLflow with S3/Glue and SageMaker for model lifecycle management."
        )

    # Additional quick checks
    if "displayHTML" in detected:
        suggestions.append(
            "displayHTML found - review any HTML/JS embedded in notebooks for unsupported browser integrations."
        )

    # Risk color
    risk = "green"
    if complexity == "complex" or any(i for i in issues):
        risk = "red"
    elif complexity == "medium":
        risk = "yellow"

    summary = (
        f"Detected {len(detected)} Databricks-specific patterns. Complexity: {complexity}."
        if detected
        else f"Complexity: {complexity}. No obvious Databricks-specific patterns detected."
    )

    return {
        "path": path,
        "language": language,
        "complexity": complexity,
        "issues": issues,
        "suggestions": suggestions,
        "detected_patterns": detected,
        "risk": risk,
        "summary": summary,
    }


def analyze_notebooks_batch(
    notebooks: List[Dict[str, Any]], config: Optional[ConfigManager] = None
) -> List[Dict[str, Any]]:
    results = []
    bedrock_enabled = False
    if config:
        try:
            bedrock_enabled = bool(
                config.get("features.enable_bedrock_analysis", False)
            )
        except Exception:
            bedrock_enabled = False

    for n in notebooks:
        content = n.get("content", "")
        path = n.get("path") or n.get("object_path") or n.get("name", "<unknown>")
        lang = n.get("language", "python")
        res = analyze_notebook(content, path, lang)
        if bedrock_enabled and config:
            prompt = (
                "You are an assistant that summarizes Databricks notebooks for migration to AWS.\n"
                f"Notebook path: {path}\n"
                "Provide: 1) short migration summary, 2) critical migration blockers, 3) suggested AWS approach.\n"
                "Notebook content (first 4000 chars):\n" + (content[:4000])
            )
            enrich = _call_bedrock_model(prompt, config)
            res["bedrock_enrichment"] = enrich
        else:
            res["bedrock_enrichment"] = {"note": "not enabled"}
        # Wrap analysis with the original notebook metadata to match expected structure
        results.append({"notebook": n, "analysis": res})
    return results
