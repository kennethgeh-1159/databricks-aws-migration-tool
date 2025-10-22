"""Bedrock-backed notebook analyzer.

Provides BedrockAnalyzer which uses boto3 'bedrock' and 'bedrock-runtime'
clients to analyze Databricks notebooks (Claude 3 Sonnet model recommended).

The analyzer includes robust error handling, request truncation to stay within
token limits, and stable JSON parsing with sensible defaults.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


class BedrockAnalyzer:
    """Analyze Databricks notebooks using Amazon Bedrock (Claude 3 Sonnet).

    Args:
        region_name: AWS region for Bedrock endpoints (defaults to AWS_REGION env)
        model_id: Bedrock model id to invoke (defaults to BEDROCK_MODEL_ID env)
    """

    DEFAULT_TRUNCATE = 8000

    def __init__(
        self, region_name: Optional[str] = None, model_id: Optional[str] = None
    ) -> None:
        self.region_name = region_name or os.getenv("AWS_REGION")
        self.model_id = model_id or os.getenv("BEDROCK_MODEL_ID")

        # Primary (control plane) client and runtime client
        try:
            self.client = (
                boto3.client("bedrock", region_name=self.region_name)
                if self.region_name
                else boto3.client("bedrock")
            )
        except Exception as e:
            logger.exception("failed to create bedrock client: %s", e)
            self.client = None

        try:
            # Some environments provide a distinct runtime client
            self.runtime_client = (
                boto3.client("bedrock-runtime", region_name=self.region_name)
                if self.region_name
                else boto3.client("bedrock-runtime")
            )
        except Exception as e:
            logger.exception("failed to create bedrock-runtime client: %s", e)
            self.runtime_client = None

    def _truncate(self, text: str, limit: Optional[int] = None) -> str:
        lim = limit or self.DEFAULT_TRUNCATE
        if len(text) <= lim:
            return text
        # Keep start and end context
        head = text[: int(lim * 0.6)]
        tail = text[-int(lim * 0.4) :]
        return head + "\n\n...TRUNCATED...\n\n" + tail

    def test_bedrock_access(self) -> Dict[str, Any]:
        """List available foundation models and check Claude accessibility.

        Returns a dictionary with 'ok' bool and 'models' list (names/ids) or 'error'.
        """
        result: Dict[str, Any] = {"ok": False, "models": [], "error": None}
        if not self.client:
            result["error"] = "Bedrock control-plane client not initialized"
            return result

        try:
            # Attempt to list models. SDKs and clients expose different names
            # (list_models, list_foundational_models, listFoundationalModels, etc.).
            # Try a sequence of common method names and use the first that exists.
            resp = None
            candidate_methods = [
                "list_models",
                "list_foundational_models",
                "list_foundation_models",
                "listFoundationalModels",
                "list_foundationalModels",
            ]

            for m in candidate_methods:
                if hasattr(self.client, m):
                    try:
                        fn = getattr(self.client, m)
                        resp = fn()
                        break
                    except Exception as ex:
                        # record and continue trying other methods
                        logger.debug("bedrock list method %s raised: %s", m, ex)
                        resp = None
                        continue

            # If no high-level method worked, try low-level operation names
            if resp is None:
                try:
                    if hasattr(self.client, "meta") and hasattr(
                        self.client.meta, "service_model"
                    ):
                        ops = getattr(
                            self.client.meta.service_model, "operation_names", []
                        )
                    else:
                        ops = []

                    for op in (
                        "ListModels",
                        "ListFoundationModels",
                        "ListFoundationalModels",
                    ):
                        if op in ops and hasattr(self.client, "_make_api_call"):
                            try:
                                resp = self.client._make_api_call(op, {})
                                break
                            except Exception as ex:
                                logger.debug("_make_api_call %s failed: %s", op, ex)
                                resp = None
                                continue
                except Exception:
                    # fall through to raising a friendly error below
                    resp = None

            if resp is None:
                raise AttributeError(
                    "Bedrock client does not expose a compatible list models operation"
                )
            models = []
            if isinstance(resp, dict):
                # look for known keys
                for key in ("models", "modelSummaries", "foundationalModels"):
                    if key in resp and isinstance(resp[key], list):
                        for m in resp[key]:
                            # m may be either str or dict
                            if isinstance(m, dict):
                                name = (
                                    m.get("modelId")
                                    or m.get("name")
                                    or m.get("id")
                                    or m.get("modelName")
                                )
                            else:
                                name = str(m)
                            if name:
                                models.append(name)
                        break

            result["ok"] = True
            result["models"] = models
            # quick heuristic to check Claude presence
            result["claude_available"] = any(
                "claude" in (m or "").lower() for m in models
            )
            return result

        except (BotoCoreError, ClientError) as e:
            logger.exception("error listing Bedrock models: %s", e)
            result["error"] = str(e)
            return result
        except Exception as e:
            logger.exception("unexpected error in test_bedrock_access: %s", e)
            result["error"] = str(e)
            return result

    def _invoke_runtime(self, prompt: str) -> Tuple[bool, Dict[str, Any]]:
        """Invoke the runtime client using self.model_id. Returns (ok, parsed_result).

        The method is resilient to differences in SDK shapes. Uses 'body' streaming
        read handling and returns parsed JSON where possible.
        """
        if not self.runtime_client:
            return False, {"error": "Bedrock runtime client not initialized"}

        # truncate prompt to avoid token limits
        truncated = self._truncate(prompt, self.DEFAULT_TRUNCATE)

        try:
            # Preferred modern signature: invoke_model(ModelId=..., Body=b'...')
            if hasattr(self.runtime_client, "invoke_model"):
                try:
                    resp = self.runtime_client.invoke_model(
                        ModelId=self.model_id, Body=truncated.encode("utf-8")
                    )
                except TypeError:
                    resp = self.runtime_client.invoke_model(
                        modelId=self.model_id, body=truncated.encode("utf-8")
                    )
            elif hasattr(self.runtime_client, "invoke"):
                try:
                    resp = self.runtime_client.invoke(
                        modelId=self.model_id, body=truncated.encode("utf-8")
                    )
                except TypeError:
                    resp = self.runtime_client.invoke(
                        ModelId=self.model_id, Body=truncated.encode("utf-8")
                    )
            else:
                # last resort: try low-level call names
                if hasattr(self.runtime_client, "_make_api_call"):
                    op_candidates = ["InvokeModel", "Invoke"]
                    last_exc = None
                    for op in op_candidates:
                        try:
                            resp = self.runtime_client._make_api_call(
                                op, {"modelId": self.model_id, "body": truncated}
                            )
                            break
                        except Exception as ex:
                            last_exc = ex
                            continue
                    else:
                        return False, {
                            "error": f"no suitable runtime op found, last_exc: {last_exc}"
                        }
                else:
                    return False, {
                        "error": "runtime client has no suitable invocation method"
                    }

            # Parse typical response shapes
            body = None
            if isinstance(resp, dict):
                body = resp.get("body") or resp.get("output") or resp.get("result")
            else:
                body = getattr(resp, "body", None) if hasattr(resp, "body") else None

            if hasattr(body, "read"):
                raw = body.read()
            else:
                raw = body

            if isinstance(raw, (bytes, bytearray)):
                decoded = raw.decode("utf-8", errors="replace")
            else:
                decoded = str(raw)

            decoded = decoded.strip()
            parsed: Dict[str, Any]
            if decoded.startswith("{") or decoded.startswith("["):
                try:
                    parsed = json.loads(decoded)
                except Exception:
                    parsed = {"text": decoded}
            else:
                parsed = {"text": decoded}

            return True, parsed

        except (BotoCoreError, ClientError) as e:
            logger.exception("AWS error invoking bedrock runtime: %s", e)
            return False, {"error": str(e)}
        except Exception as e:
            logger.exception("unexpected error invoking bedrock runtime: %s", e)
            return False, {"error": str(e)}

    def analyze_notebook_complexity(
        self, content: str, path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze notebook content using Claude 3 Sonnet via Bedrock runtime.

        Returns a structured dict with keys:
            - complexity_score (1-10)
            - risk_assessment
            - migration_challenges
            - databricks_specific_features
            - estimated_effort_hours
            - aws_service_recommendations

        This method truncates content to the configured token limit and handles
        errors gracefully, returning default values where parsing fails.
        """
        prompt = (
            "You are an expert migration analyst helping migrate Databricks notebooks to AWS.\n"
            "Analyze the notebook for Databricks-specific features such as dbutils, display(), Delta Lake, MLflow, widgets, and magic commands.\n"
            "For the provided notebook content, return a JSON object with the following fields:\n"
            "  - complexity_score: integer 1-10 (10 = most complex)\n"
            "  - risk_assessment: one of [low, medium, high]\n"
            "  - migration_challenges: list of short strings describing blockers\n"
            "  - databricks_specific_features: list of detected features\n"
            "  - estimated_effort_hours: number (estimate of engineering hours)\n"
            "  - aws_service_recommendations: list of {service:reason} entries\n"
            "Return only the JSON object and nothing else.\n\n"
            "Notebook content:\n" + (self._truncate(content, self.DEFAULT_TRUNCATE))
        )

        ok, parsed = self._invoke_runtime(prompt)
        # Default fallback
        defaults = {
            "complexity_score": 3,
            "risk_assessment": "medium",
            "migration_challenges": [],
            "databricks_specific_features": [],
            "estimated_effort_hours": 8,
            "aws_service_recommendations": [],
        }

        if not ok:
            # Return defaults plus the error
            defaults["error"] = (
                parsed.get("error") if isinstance(parsed, dict) else str(parsed)
            )
            return defaults

        # Try to find a JSON object in parsed response
        try:
            # parsed may be nested under different keys depending on shape
            candidate = None
            if isinstance(parsed, dict):
                # common keys
                for k in ("result", "output", "body", "Response", "content"):
                    if k in parsed and isinstance(parsed[k], (dict, str)):
                        candidate = parsed[k]
                        break
                if candidate is None:
                    # If parsed directly has expected fields, use it
                    candidate = parsed

            # If candidate is a string, try to JSON decode
            if isinstance(candidate, str):
                try:
                    candidate = json.loads(candidate)
                except Exception:
                    candidate = None

            if isinstance(candidate, dict):
                out = {
                    "complexity_score": int(
                        candidate.get("complexity_score")
                        or candidate.get("complexity")
                        or defaults["complexity_score"]
                    ),
                    "risk_assessment": candidate.get("risk_assessment")
                    or candidate.get("risk")
                    or defaults["risk_assessment"],
                    "migration_challenges": candidate.get("migration_challenges")
                    or candidate.get("challenges")
                    or defaults["migration_challenges"],
                    "databricks_specific_features": candidate.get(
                        "databricks_specific_features"
                    )
                    or candidate.get("features")
                    or defaults["databricks_specific_features"],
                    "estimated_effort_hours": float(
                        candidate.get("estimated_effort_hours")
                        or candidate.get("effort_hours")
                        or defaults["estimated_effort_hours"]
                    ),
                    "aws_service_recommendations": candidate.get(
                        "aws_service_recommendations"
                    )
                    or candidate.get("recommendations")
                    or defaults["aws_service_recommendations"],
                }
                return out

        except Exception as e:
            logger.exception("failed to parse bedrock response: %s", e)

        # If we couldn't parse, return defaults but include the raw text
        try:
            defaults["raw_response_preview"] = str(parsed)[:200]
        except Exception:
            defaults["raw_response_preview"] = None

        return defaults
