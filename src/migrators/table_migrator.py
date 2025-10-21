"""Table migration helpers.

Generate JSON sidecars with schema metadata and Databricks copy/export notebooks
that can be executed inside Databricks to move data from DBFS to S3. Optionally
upload artifacts to S3 if boto3 is available and credentials exist.
"""

import json
import logging
from typing import Dict, Optional

try:
    import boto3
    from botocore.exceptions import ClientError
except Exception:
    boto3 = None

    class ClientError(Exception):
        pass


logger = logging.getLogger(__name__)


class TableMigrator:
    """Generate migration artifacts for a table."""

    def __init__(self, config: Dict):
        self.config = config
        self.default_bucket = (
            config.get("aws", {}).get("s3", {}).get("bucket")
            or config.get("aws", {}).get("default_bucket")
            or "migrated-bucket"
        )
        if boto3:
            try:
                self.s3 = boto3.client(
                    self.config.get("aws", {}).get("s3_client_name", "s3")
                )
            except Exception:
                self.s3 = None
        else:
            self.s3 = None

    def _safe_table_name_prefix(self, table_info: Dict) -> str:
        # Build a workspace path-like prefix from catalog/schema/table
        cat = table_info.get("catalog") or "default"
        sch = table_info.get("schema") or table_info.get("db") or "default"
        name = table_info.get("name") or table_info.get("table_name") or "table"
        return f"{cat}/{sch}/{name}"

    def generate_sidecar(self, table_info: Dict) -> Dict:
        """Create a JSON-serializable sidecar with table schema metadata."""
        sidecar = {
            "name": table_info.get("name"),
            "catalog": table_info.get("catalog"),
            "schema": table_info.get("schema"),
            "table_type": table_info.get("table_type"),
            "storage_location": table_info.get("storage_location"),
            "size_bytes": table_info.get("size_bytes"),
            "created_at": table_info.get("created_at"),
            "updated_at": table_info.get("updated_at"),
            "owner": table_info.get("owner"),
            "columns": table_info.get("columns") or [],
            "partitions": table_info.get("partitions") or [],
            "notes": table_info.get("comment") or "",
        }
        return sidecar

    def generate_copy_notebook(
        self,
        table_info: Dict,
        bucket: Optional[str] = None,
        keep_delta: bool = True,
        use_copy_files: bool = False,
    ) -> str:
        """Generate a Python Databricks notebook (source) that copies the table to S3.

        If use_copy_files=True the notebook will perform dbutils.fs.cp from the
        storage_location to the target S3 path. Otherwise it will read via spark
        and write using spark.write (keeps schema).
        """
        bucket = bucket or self.default_bucket
        prefix = self._safe_table_name_prefix(table_info)
        dest_path = f"s3://{bucket}/migrated-tables/{prefix}"

        storage_location = (
            table_info.get("storage_location") or table_info.get("location") or ""
        )

        if use_copy_files and storage_location:
            # Use dbutils.fs.cp for raw file copy from DBFS to S3
            nb = f"""
# Databricks notebook: Copy table files from DBFS to S3
src = '{storage_location}'
dst = '{dest_path}'
print(f'Copying from {{src}} to {{dst}}')
try:
    dbutils.fs.cp(src, dst, recurse=True)
    print('Copy complete')
except Exception as e:
    print(f'Copy failed: {{e}}')
"""
            return nb

        # Default: use Spark to read and write the table
        table_ref = (
            table_info.get("full_name")
            or f"{table_info.get('catalog')}.{table_info.get('schema')}.{table_info.get('name')}"
        )
        fmt = "delta" if keep_delta else "parquet"

        nb = f"""
# Databricks notebook: Export table to S3 using Spark
print('Exporting table: {table_ref} to {dest_path} as {fmt}')
try:
    # Try reading by table reference first, fallback to storage location
    try:
        df = spark.read.table("{table_ref}")
    except Exception:
        df = spark.read.format('delta').load('{storage_location}')

    df.write.format('{fmt}').mode('overwrite').option('overwriteSchema', 'true').save('{dest_path}')
    print('Export complete')
except Exception as e:
    print(f'Export failed: {{e}}')
"""

        return nb

    def upload_artifact(
        self, content: bytes, key: str, bucket: Optional[str] = None
    ) -> bool:
        bucket = bucket or self.default_bucket
        if not self.s3:
            logger.warning(
                "boto3 not available or s3 client not initialized; upload skipped"
            )
            return False
        try:
            self.s3.put_object(Bucket=bucket, Key=key, Body=content)
            return True
        except ClientError as e:
            logger.error(f"Failed to upload artifact to s3://{bucket}/{key}: {e}")
            return False
        except Exception as e:
            try:
                from botocore.exceptions import (
                    NoCredentialsError,
                    PartialCredentialsError,
                )

                if isinstance(e, (NoCredentialsError, PartialCredentialsError)):
                    logger.warning("AWS credentials not found; upload skipped: %s", e)
                    return False
            except Exception:
                if "Unable to locate credentials" in str(
                    e
                ) or "Could not find credentials" in str(e):
                    logger.warning("AWS credentials not found; upload skipped: %s", e)
                    return False

            logger.exception("Failed to upload artifact to S3: %s", e)
            return False
