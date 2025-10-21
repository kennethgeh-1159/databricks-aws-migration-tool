"""Notebook analysis utilities.

Provides heuristic analysis for Databricks notebooks to identify
Databricks-specific syntax, estimate complexity, flag issues, and
suggest AWS equivalent approaches. Optionally calls Amazon Bedrock
when enabled in config (features.enable_bedrock_analysis). Bedrock
enrichment is best-effort and failures are returned in the
`bedrock_enrichment` field without failing the analysis pipeline.

This module implements a robust _call_bedrock_model that attempts
multiple invocation shapes and writes a small diagnostic file
(`data/results/bedrock_client_diag.json`) describing the client
surface when a call cannot be performed. The diagnostic file does
NOT include credentials or secret values.
"""

import os
from typing import Any, Dict, List, Optional

"""Notebook analysis utilities.

Provides heuristic analysis for Databricks notebooks to identify
Databricks-specific syntax, estimate complexity, flag issues, and
suggest AWS equivalent approaches. Optionally calls Amazon Bedrock
when enabled in config (features.enable_bedrock_analysis). Bedrock
enrichment is best-effort and failures are returned in the
`bedrock_enrichment` field without failing the analysis pipeline.

This module implements a robust _call_bedrock_model that attempts
multiple invocation shapes and writes a small diagnostic file
(`data/results/bedrock_client_diag.json`) describing the client
surface when a call cannot be performed. The diagnostic file does
NOT include credentials or secret values.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Simple pattern map for Databricks-specific constructs
DB_SPECIFIC_PATTERNS = {
    "dbutils": r"\bdbutils\b",
    "display": r"\bdisplay\b",
    "displayHTML": r"\bdisplayHTML\b",
    "sql_cell_magic": r"^%sql",
    "pyspark": r"\bspark\.(read|sql|createDataFrame)\b",
    "delta": r"\bdelta\.tables\b|\bDeltaTable\b",
}


# Minimal ConfigManager typing hint to avoid importing heavy objects at module import
class ConfigManagerLike(Dict):
    pass


def _write_bedrock_diag(diag: Dict[str, Any]):
    """Write a small diagnostic file with client introspection (no secrets).

    This helps debugging when users report an incompatible Bedrock client
    surface. We deliberately only write attribute names and types; we do not
    log credential values.
    """
    try:
        os.makedirs("data/results", exist_ok=True)
        path = os.path.join("data/results", "bedrock_client_diag.json")
        with open(path, "w") as f:
            json.dump(diag, f, indent=2)
    except Exception:
        logger.exception("failed to write bedrock diagnostic file")


def _call_bedrock_model(
    prompt: str,
    config: Optional[ConfigManagerLike] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Best-effort call to Amazon Bedrock via boto3.

    This function attempts multiple invocation styles to handle different
    boto3/bedrock client variants. On failure it returns a structured error
    dictionary instead of raising so the analysis pipeline remains resilient.

    It also writes a diagnostic file with the client's attribute list when
    a call cannot be made so we can adapt to user environments.
    """
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, PartialCredentialsError
    except Exception as e:
        return {"error": f"boto3 not available: {e}", "raw": None}

    # Resolve configured model id
    bedrock_cfg = (
        config.get("aws", {}).get("bedrock", {}) if isinstance(config, dict) else {}
    )
    configured_model = model_id or (
        bedrock_cfg.get("model_id") if isinstance(bedrock_cfg, dict) else None
    )
    if not configured_model:
        return {
            "error": "Bedrock model_id not configured in aws.bedrock.model_id",
            "raw": None,
        }

    # determine region
    region = (
        (config.get("aws", {}).get("region") if isinstance(config, dict) else None)
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
    )

    try:
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

    # Introspect client shape for diagnostics (do not include sensitive values)
    try:
        client_type = type(client).__name__
        client_module = type(client).__module__
        attrs = sorted(a for a in dir(client) if not a.startswith("__"))[:300]
        diag = {"type": client_type, "module": client_module, "attributes": attrs}
        _write_bedrock_diag(diag)
    except Exception:
        logger.exception("failed to introspect bedrock client")

    # Try direct methods first
    try:
        if hasattr(client, "invoke_model"):
            fn = getattr(client, "invoke_model")
            try:
                response = fn(modelId=configured_model, body=prompt.encode("utf-8"))
            except TypeError:
                response = fn(ModelId=configured_model, Body=prompt.encode("utf-8"))
        elif hasattr(client, "invoke"):
            fn = getattr(client, "invoke")
            try:
                response = fn(modelId=configured_model, body=prompt.encode("utf-8"))
            except TypeError:
                response = fn(ModelId=configured_model, Body=prompt.encode("utf-8"))
        else:
            # Try low-level _make_api_call with common op names
            if hasattr(client, "meta") and hasattr(client.meta, "service_model"):
                op_names = list(
                    getattr(client.meta.service_model, "operation_names", [])
                )
            else:
                op_names = []

            candidate_ops = [
                op
                for op in ("InvokeModel", "Invoke", "invoke_model", "invoke")
                if op in op_names
            ]
            if hasattr(client, "_make_api_call") and candidate_ops:
                last_exc = None
                for op in candidate_ops:
                    for params in (
                        {"modelId": configured_model, "body": prompt.encode("utf-8")},
                        {"ModelId": configured_model, "Body": prompt.encode("utf-8")},
                        {"modelId": configured_model, "body": prompt},
                        {"ModelId": configured_model, "Body": prompt},
                    ):
                        try:
                            response = client._make_api_call(op, params)
                            raise StopIteration
                        except StopIteration:
                            break
                        except Exception as e:
                            last_exc = e
                            continue
                    else:
                        continue
                    break
                # If we exhausted loops without setting response, fail
                try:
                    response  # type: ignore
                except NameError:
                    return {"error": f"bedrock call failed: {last_exc}", "raw": None}
            else:
                return {
                    "error": "Bedrock client does not expose invoke_model or invoke",
                    "raw": None,
                }

    except Exception as e:
        return {"error": f"bedrock call failed: {e}", "raw": None}

    # Parse response
    try:
        body = (
            response.get("body")
            if isinstance(response, dict)
            else getattr(response, "body", None)
        )
        if hasattr(body, "read"):
            raw = body.read()
        else:
            raw = body

        if isinstance(raw, (bytes, bytearray)):
            decoded = raw.decode("utf-8", errors="replace")
        else:
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

    Returns a dict with: complexity, issues, suggestions, detected_patterns, risk, summary
    """
    issues: List[str] = []
    suggestions: List[str] = []
    detected: List[str] = []

    for name, pat in DB_SPECIFIC_PATTERNS.items():
        try:
            if re.search(pat, content, flags=re.MULTILINE):
                detected.append(name)
        except re.error:
            continue

    loc = len(content.splitlines())
    complexity = "simple"
    if loc < 100:
        complexity = "simple"
    elif loc < 500:
        complexity = "medium"
    else:
        complexity = "complex"

    if "dbutils" in detected or re.search(r"\bdbutils\.", content):
        issues.append("Uses dbutils - these utilities are Databricks-specific.")
        suggestions.append("Replace dbutils calls with boto3 or AWS equivalents.")

    if "display" in detected or re.search(r"\bdisplay\(", content):
        suggestions.append(
            "Uses display() - consider rendering in alternative AWS tooling."
        )

    risk = "green"
    if complexity == "complex" or issues:
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
    notebooks: List[Dict[str, Any]], config: Optional[ConfigManagerLike] = None
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    bedrock_enabled = False
    try:
        if config and isinstance(config, dict):
            bedrock_enabled = bool(
                config.get("features", {}).get("enable_bedrock_analysis", False)
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

        results.append({"notebook": n, "analysis": res})

    return results
