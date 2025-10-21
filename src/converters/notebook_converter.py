"""
Deterministic, template-based notebook converter for Databricks -> AWS migration.

This module provides a simple, auditable transformation that:
- Accepts notebook source text (PY/SCALA/SQL as plain source exported by Databricks)
- Applies deterministic string replacements for common Databricks idioms
- Injects S3 helper functions (boto3 wrappers) at the top of converted Python/Scala files
- Uploads converted files to S3 under a structured prefix
- Produces a before/after diff report for each converted file

This converter intentionally avoids LLMs or heuristics; it's rule-based and reversible.
"""

import io
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

try:
    import boto3
    from botocore.exceptions import ClientError
except Exception:
    boto3 = None

    # Use a generic Exception fallback so code referencing ClientError still works
    class ClientError(Exception):
        pass


from utils.base import BaseComponent, MigrationResult

logger = logging.getLogger(__name__)


S3_HELPER_TEMPLATE_PY = """
# --- S3 helper functions injected by databricks-aws-migration-tool ---
import boto3
import os
from botocore.exceptions import ClientError
_s3 = boto3.client('s3')

def upload_file_to_s3(file_bytes: bytes, bucket: str, key: str) -> bool:
    try:
        _s3.put_object(Bucket=bucket, Key=key, Body=file_bytes)
        return True
    except ClientError as e:
        print(f"Failed to upload to s3://{bucket}/{key}: {e}")
        return False

# End helpers
"""

# Add more complete helpers for list/copy so converted notebooks have runnable stubs
S3_HELPER_TEMPLATE_PY = (
    S3_HELPER_TEMPLATE_PY
    + '''

def s3_list(s3_path: str):
    """List objects under an s3://bucket/prefix path.

    Returns a list of keys (not full s3:// URIs).
    """
    try:
        if not s3_path.startswith('s3://'):
            raise ValueError('s3_list expects s3:// path')
        parts = s3_path[5:].split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''
        paginator = _s3.get_paginator('list_objects_v2')
        keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                keys.append(obj.get('Key'))
        return keys
    except Exception as e:
        print(f's3_list error: {e}')
        return []


def s3_copy(src_s3: str, dest_s3: str):
    """Copy object from s3://src_bucket/src_key to s3://dest_bucket/dest_key.

    Returns True on success.
    """
    try:
        if not src_s3.startswith('s3://') or not dest_s3.startswith('s3://'):
            raise ValueError('s3_copy expects s3:// URIs')
        s_src = src_s3[5:].split('/', 1)
        s_dest = dest_s3[5:].split('/', 1)
        copy_source = {'Bucket': s_src[0], 'Key': s_src[1]}
        _s3.copy(copy_source, s_dest[0], s_dest[1])
        return True
    except Exception as e:
        print(f's3_copy error: {e}')
        return False

'''
)

S3_HELPER_TEMPLATE_SCALA = """
// --- S3 helper functions injected by databricks-aws-migration-tool ---
// Uses AWS Java SDK or Hadoop S3A paths; keep simple: write to local FS then use aws cli or s3a
// (Conversion: users should adapt this for production Scala jobs)
"""


class NotebookConverter(BaseComponent):
    """Converter that performs deterministic replacements and S3 uploads."""

    REPLACEMENTS = [
        # (pattern, replacement, flags)
        (r"\bdisplay\s*\(", "print(", 0),
        (r"\bdisplayHTML\s*\(", "print(", 0),
        # dbutils.fs.ls => list s3 objects via boto3: we replace call site textually; full runtime
        # conversion requires manual review. We convert common "dbutils.fs.ls('/path')" -> "list_s3('s3://bucket/...')"
        (r"dbutils\.fs\.ls\s*\(", "s3_list(", 0),
        (r"dbutils\.fs\.cp\s*\(", "s3_copy(", 0),
        (r"spark\.read\.format\(\s*\"delta\"\s*\)", 'spark.read.format("parquet")', 0),
        (r"/dbfs/", "s3://{bucket}/", 0),
        (r"dbfs:/", "s3://{bucket}/", 0),
    ]

    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        # Initialize boto3 client lazily; allow import even when boto3 isn't installed
        if boto3:
            try:
                self.s3 = boto3.client(
                    self.get_config_value("aws.s3_client_name", "s3")
                )
            except Exception:
                self.s3 = None
                self.logger.warning(
                    "Failed to initialize boto3 S3 client; uploads will be disabled."
                )
        else:
            self.s3 = None
            self.logger.warning(
                "boto3 not installed: S3 uploads are disabled. Install boto3 to enable uploads."
            )
        self.bucket_default = self.get_config_value(
            "aws.default_bucket", "migrated-bucket"
        )

    def validate_config(self) -> bool:
        """Validate that required AWS config is present for uploads.

        Returns True if configuration is acceptable (we treat missing bucket as warning).
        """
        bucket = self.get_config_value("aws.default_bucket") or self.get_config_value(
            "aws.s3.bucket"
        )
        if not bucket:
            # Do not raise — allow dry-run conversions, but warn the user
            self.logger.warning(
                "NotebookConverter: No S3 bucket configured (aws.default_bucket or aws.s3.bucket). Uploads will fail unless a bucket is provided."
            )
        return True

    def _apply_replacements(
        self, source: str, bucket: Optional[str] = None, language: str = "PYTHON"
    ) -> str:
        bucket = bucket or self.bucket_default
        out = source
        for pattern, replacement, flags in self.REPLACEMENTS:
            repl = replacement.format(bucket=bucket)
            if flags:
                out = re.sub(pattern, repl, out, flags=flags)
            else:
                out = re.sub(pattern, repl, out)
        return out

    def _inject_helpers(self, source: str, language: str = "PYTHON") -> str:
        if language and language.upper() in ("PYTHON", "PY"):
            return S3_HELPER_TEMPLATE_PY + "\n" + source
        elif language and language.upper() in ("SCALA",):
            return S3_HELPER_TEMPLATE_SCALA + "\n" + source
        else:
            return source

    def _upload_converted(
        self, converted_bytes: bytes, dest_key: str, bucket: Optional[str] = None
    ) -> bool:
        bucket = bucket or self.bucket_default
        if not bucket:
            logger.warning("No S3 bucket configured: skipping upload")
            return False
        if not self.s3:
            logger.warning("boto3 not available: skipping upload")
            return False
        try:
            self.s3.put_object(Bucket=bucket, Key=dest_key, Body=converted_bytes)
            logger.info("Uploaded converted notebook to s3://%s/%s", bucket, dest_key)
            return True
        except Exception as e:
            # Handle common credentials errors gracefully so the app can run in dry-run mode
            try:
                from botocore.exceptions import (
                    NoCredentialsError,
                    PartialCredentialsError,
                )

                if isinstance(e, (NoCredentialsError, PartialCredentialsError)):
                    logger.warning("AWS credentials not found; upload skipped: %s", e)
                    return False
            except Exception:
                # botocore may not be available in minimal dev environments
                if "Unable to locate credentials" in str(
                    e
                ) or "Could not find credentials" in str(e):
                    logger.warning("AWS credentials not found; upload skipped: %s", e)
                    return False

            logger.exception("Failed to upload converted notebook to S3: %s", e)
            return False

    def convert_notebook(
        self,
        source: str,
        path: str,
        language: str = "PYTHON",
        bucket: Optional[str] = None,
    ) -> Dict:
        """Convert a single notebook source and upload to S3.

        Returns a dict with keys: path, converted_key, uploaded (bool), before, after, diffs (list)
        """
        bucket = bucket or self.bucket_default
        before = source
        converted = self._apply_replacements(source, bucket=bucket, language=language)
        converted = self._inject_helpers(converted, language=language)

        # produce a simple line-based unified diff
        before_lines = before.splitlines(keepends=True)
        after_lines = converted.splitlines(keepends=True)
        diffs = list(self._simple_line_diff(before_lines, after_lines))

        # construct destination key
        safe_path = path.lstrip("/")
        dest_key = f"migrated-notebooks/{safe_path}"

        uploaded = self._upload_converted(
            converted.encode("utf-8"), dest_key, bucket=bucket
        )

        return {
            "path": path,
            "converted_key": dest_key,
            "uploaded": uploaded,
            "before_lines": before_lines,
            "after_lines": after_lines,
            "diffs": diffs,
        }

    def convert_notebooks_batch(
        self, notebooks: List[Dict], bucket: Optional[str] = None
    ) -> List[Dict]:
        """Convert and upload a batch of notebooks (notebook comes as dict with path, content, language)."""
        results = []
        for n in notebooks:
            try:
                src = n.get("content") or ""
                path = n.get("path") or n.get("name") or "unnamed.py"
                lang = n.get("language") or "PYTHON"
                res = self.convert_notebook(src, path, language=lang, bucket=bucket)
                results.append(res)
            except Exception as e:
                self.logger.exception(
                    f"Failed to convert notebook {n.get('path')}: {e}"
                )
                results.append({"path": n.get("path"), "error": str(e)})
        return results

    def _simple_line_diff(self, a_lines: List[str], b_lines: List[str]):
        """Yield a simple line-based diff tuples: (op, line). op = ' ' (same), '+' (added), '-' (removed)"""
        import difflib

        seq = difflib.SequenceMatcher(a=a_lines, b=b_lines)
        for tag, i1, i2, j1, j2 in seq.get_opcodes():
            if tag == "equal":
                for line in a_lines[i1:i2]:
                    yield (" ", line)
            elif tag == "replace":
                for line in a_lines[i1:i2]:
                    yield ("-", line)
                for line in b_lines[j1:j2]:
                    yield ("+", line)
            elif tag == "delete":
                for line in a_lines[i1:i2]:
                    yield ("-", line)
            elif tag == "insert":
                for line in b_lines[j1:j2]:
                    yield ("+", line)
