"""
Databricks API Connector for the migration tool.
Handles authentication and data extraction from Databricks workspace.
"""

import base64
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from utils.base import BaseComponent, MigrationResult


class DatabricksConnector(BaseComponent):
    """Connects to Databricks workspace and extracts notebooks and tables."""

    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize Databricks connector.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        super().__init__(config, logger)

        # Initialize session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.get_config_value("databricks.max_retries", 3),
            backoff_factor=self.get_config_value("databricks.retry_delay_seconds", 1),
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _setup(self) -> None:
        """Setup Databricks-specific configuration."""
        workspace_url = self.get_config_value("databricks.workspace.url")
        access_token = self.get_config_value("databricks.workspace.token")

        if not workspace_url or not access_token:
            raise ValueError("Databricks workspace URL and access token are required")

        self.workspace_url = workspace_url.rstrip("/")
        self.base_url = f"https://{self.workspace_url}/api/{self.get_config_value('databricks.api_version', '2.0')}"

        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": "DatabricksMigrationTool/1.0.0",
        }

        self.timeout = self.get_config_value("databricks.timeout_seconds", 60)

    def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Unified request helper with retries, exponential backoff and graceful handling of
        read timeouts and connection errors. Returns a requests.Response or None on final failure.

        Configuration keys respected:
        - databricks.max_retries
        - databricks.retry_delay_seconds
        - databricks.timeout_seconds
        """
        headers = headers or self.headers
        max_attempts = int(self.get_config_value("databricks.max_retries", 3))
        base_delay = float(self.get_config_value("databricks.retry_delay_seconds", 1))

        for attempt in range(1, max_attempts + 1):
            try:
                resp = self.session.request(
                    method,
                    url,
                    params=params,
                    json=data,
                    headers=headers,
                    timeout=self.timeout,
                    **kwargs,
                )
                return resp

            except requests.exceptions.ReadTimeout as e:
                self.logger.warning(
                    f"ReadTimeout on attempt {attempt}/{max_attempts} for {url}: {e}"
                )
            except requests.exceptions.ConnectTimeout as e:
                self.logger.warning(
                    f"ConnectTimeout on attempt {attempt}/{max_attempts} for {url}: {e}"
                )
            except requests.exceptions.ConnectionError as e:
                self.logger.warning(
                    f"ConnectionError on attempt {attempt}/{max_attempts} for {url}: {e}"
                )
            except requests.exceptions.RequestException as e:
                # Non-retriable or unexpected request-level error; log and abort
                self.logger.error(f"RequestException for {url}: {e}")
                return None

            # If we get here, we will retry unless this was the last attempt
            if attempt == max_attempts:
                self.logger.error(f"Exceeded retries for {url}")
                return None

            # Exponential backoff before next attempt
            delay = base_delay * (2 ** (attempt - 1))
            self.logger.debug(f"Sleeping {delay}s before retrying {url}")
            time.sleep(delay)

    def _format_timestamp(self, value):
        """Normalize various timestamp representations into ISO8601 string or None.

        Accepts integer epoch (seconds or milliseconds), float, or ISO/date string.
        Returns ISO formatted UTC string like 'YYYY-MM-DDTHH:MM:SSZ' or None if unparseable.
        """
        if value is None:
            return None

        try:
            # numeric-like strings or numbers
            if isinstance(value, str) and value.isdigit():
                value_num = int(value)
            elif isinstance(value, (int, float)):
                value_num = int(value)
            else:
                # Try parsing strings via pandas if available
                try:
                    import pandas as pd

                    ts = pd.to_datetime(value, utc=True, errors="coerce")
                    if pd.notna(ts):
                        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    pass
                return str(value)

            # Determine seconds vs milliseconds
            if value_num > 10**12:
                ts_sec = value_num / 1000.0
            elif value_num > 10**9:
                ts_sec = value_num / 1000.0
            else:
                ts_sec = float(value_num)

            from datetime import datetime, timezone

            dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            try:
                return str(value)
            except Exception:
                return None

    def validate_config(self) -> bool:
        """Validate Databricks configuration."""
        required_fields = ["databricks.workspace.url", "databricks.workspace.token"]

        for field in required_fields:
            if not self.get_config_value(field):
                self.logger.error(f"Missing required configuration: {field}")
                return False

        return True

    def test_connection(self) -> MigrationResult:
        """
        Test connection to Databricks workspace.

        Returns:
            MigrationResult with connection status
        """
        try:
            self.logger.info("Testing Databricks connection...")

            response = self._request(
                "GET", f"{self.base_url}/clusters/list", headers=self.headers
            )

            if response.status_code == 200:
                self.logger.info("Databricks connection test successful")
                return MigrationResult(
                    status="success",
                    message="Successfully connected to Databricks workspace",
                    data={"workspace_url": self.workspace_url},
                )
            else:
                error_msg = f"Connection test failed with status {response.status_code}"
                self.logger.error(error_msg)
                return MigrationResult(
                    status="failed",
                    message=error_msg,
                    errors=[f"HTTP {response.status_code}: {response.text}"],
                )

        except requests.exceptions.RequestException as e:
            error_msg = f"Connection test failed: {str(e)}"
            self.logger.error(error_msg)
            return MigrationResult(status="failed", message=error_msg, errors=[str(e)])

    def list_notebooks(
        self, path: str = "/", recursive: bool = True
    ) -> MigrationResult:
        """
        List all notebooks in the workspace.

        Args:
            path: Starting path to search for notebooks
            recursive: Whether to search recursively in subdirectories

        Returns:
            MigrationResult containing list of notebooks
        """
        try:
            self.logger.info(f"Listing notebooks from path: {path}")
            notebooks = []

            if recursive:
                notebooks = self._list_notebooks_recursive(path)
            else:
                notebooks = self._list_notebooks_single_path(path)

                self.logger.info(f"Found {len(notebooks)} notebooks")

            return MigrationResult(
                status="success",
                message=f"Successfully listed {len(notebooks)} notebooks",
                data={"notebooks": notebooks, "count": len(notebooks)},
            )

        except Exception as e:
            error_msg = f"Error listing notebooks: {str(e)}"
            self.logger.error(error_msg)
            return MigrationResult(status="failed", message=error_msg, errors=[str(e)])

    def _list_notebooks_recursive(self, path: str) -> List[Dict]:
        """Recursively list notebooks from a path."""
        notebooks = []
        paths_to_process = [path]

        while paths_to_process:
            current_path = paths_to_process.pop(0)

            try:
                response = self._request(
                    "GET",
                    f"{self.base_url}/workspace/list",
                    params={"path": current_path, "fmt": "SOURCE"},
                    headers=self.headers,
                )

                if response and response.status_code == 200:
                    objects = response.json().get("objects", [])

                    for item in objects:
                        if item["object_type"] == "NOTEBOOK":
                            # Get notebook content
                            content = self._get_notebook_content(item["path"])

                            if content:
                                # Try to capture owner metadata if present. Databricks
                                # workspace list responses sometimes include owner or
                                # object_meta; otherwise we can optionally call
                                # workspace/get-status to retrieve more metadata.
                                owner = item.get("owner") or (
                                    item.get("object_meta") or {}
                                ).get("owner")
                                if not owner:
                                    # Honor a configuration flag to avoid extra API
                                    # calls unless explicitly enabled.
                                    try:
                                        if self.get_config_value(
                                            "databricks.fetch_notebook_owner_details",
                                            False,
                                        ):
                                            owner = self._get_notebook_owner(
                                                item["path"]
                                            )
                                    except Exception:
                                        owner = None

                                # Normalize created/modified timestamps where possible
                                raw_modified = (
                                    item.get("modified_at")
                                    or item.get("updated_at")
                                    or item.get("modified")
                                    or item.get("last_modified")
                                    or item.get("created_at")
                                    or item.get("created_time")
                                )
                                raw_created = item.get("created_at") or item.get(
                                    "created_time"
                                )

                                notebook_info = {
                                    "path": item["path"],
                                    "language": item.get("language", "PYTHON"),
                                    "content": content,
                                    "size": len(content),
                                    "object_id": item.get("object_id"),
                                    "created_at": self._format_timestamp(raw_created),
                                    "modified_at": self._format_timestamp(raw_modified),
                                    "owner": owner or "",
                                }
                                notebooks.append(notebook_info)
                                self.logger.debug(f"Added notebook: {item['path']}")

                        elif item["object_type"] == "DIRECTORY":
                            # Add directory to processing queue
                            paths_to_process.append(item["path"])
                            self.logger.debug(
                                f"Added directory to queue: {item['path']}"
                            )

                else:
                    if response is None:
                        self.logger.warning(
                            f"Failed to list path {current_path}: no response"
                        )
                    else:
                        self.logger.warning(
                            f"Failed to list path {current_path}: {response.status_code}"
                        )

            except Exception as e:
                self.logger.error(f"Error processing path {current_path}: {e}")
                continue

        return notebooks

    def _list_notebooks_single_path(self, path: str) -> List[Dict]:
        """List notebooks from a single path (non-recursive)."""
        notebooks = []

        response = self._request(
            "GET",
            f"{self.base_url}/workspace/list",
            params={"path": path, "fmt": "SOURCE"},
            headers=self.headers,
        )
        if response and response.status_code == 200:
            objects = response.json().get("objects", [])

            for item in objects:
                if item["object_type"] == "NOTEBOOK":
                    content = self._get_notebook_content(item["path"])

                    if content:
                        owner = item.get("owner") or (
                            item.get("object_meta") or {}
                        ).get("owner")
                        if not owner:
                            try:
                                if self.get_config_value(
                                    "databricks.fetch_notebook_owner_details", False
                                ):
                                    owner = self._get_notebook_owner(item["path"])
                            except Exception:
                                owner = None

                        raw_modified = (
                            item.get("modified_at")
                            or item.get("updated_at")
                            or item.get("modified")
                            or item.get("last_modified")
                            or item.get("created_at")
                            or item.get("created_time")
                        )
                        raw_created = item.get("created_at") or item.get("created_time")

                        notebook_info = {
                            "path": item["path"],
                            "language": item.get("language", "PYTHON"),
                            "content": content,
                            "size": len(content),
                            "object_id": item.get("object_id"),
                            "created_at": self._format_timestamp(raw_created),
                            "modified_at": self._format_timestamp(raw_modified),
                            "owner": owner or "",
                        }
                        notebooks.append(notebook_info)

        return notebooks

    def _get_notebook_owner(self, notebook_path: str) -> Optional[str]:
        """Attempt to fetch notebook owner via workspace/get-status.

        This is optional and controlled via `databricks.fetch_notebook_owner_details`
        in the configuration to avoid extra API calls for large workspaces.
        """
        try:
            response = self._request(
                "GET",
                f"{self.base_url}/workspace/get-status",
                params={"path": notebook_path},
                headers=self.headers,
            )
            if response and response.status_code == 200:
                body = response.json()
                # Different Databricks versions/editions expose metadata
                # under different keys; try common locations.
                owner = body.get("owner") or (body.get("object_meta") or {}).get(
                    "owner"
                )
                # Some responses include created_by/displayName
                if not owner:
                    owner = (
                        (body.get("object_meta") or {}).get("created_by")
                        or body.get("created_by")
                        or (body.get("object_meta") or {}).get("created_by_user")
                    )
                return owner
        except Exception as e:
            self.logger.debug(f"Failed to fetch owner for {notebook_path}: {e}")

        return None

    def _get_notebook_content(self, notebook_path: str) -> Optional[str]:
        """
        Get content of a specific notebook.

        Args:
            notebook_path: Path to the notebook

        Returns:
            Notebook content as string, or None if failed
        """
        try:
            response = self._request(
                "GET",
                f"{self.base_url}/workspace/export",
                params={"path": notebook_path, "format": "SOURCE"},
                headers=self.headers,
            )

            if response and response.status_code == 200:
                content_b64 = response.json().get("content", "")
                if content_b64:
                    # Decode base64 content
                    decoded_content = base64.b64decode(content_b64).decode("utf-8")
                    return decoded_content
            else:
                if response is None:
                    self.logger.warning(
                        f"Failed to get content for {notebook_path}: no response"
                    )
                else:
                    self.logger.warning(
                        f"Failed to get content for {notebook_path}: {response.status_code}"
                    )

        except Exception as e:
            self.logger.error(
                f"Error getting notebook content for {notebook_path}: {e}"
            )

        return None

    def list_tables(
        self, catalog: Optional[str] = None, schema: Optional[str] = None
    ) -> MigrationResult:
        """
        List all tables in the workspace.

        Args:
            catalog: Optional catalog filter
            schema: Optional schema filter

        Returns:
            MigrationResult containing list of tables
        """
        try:
            self.logger.info("Listing tables from workspace")
            tables = []

            # Try Unity Catalog first
            unity_tables = self._list_unity_catalog_tables(catalog, schema)
            if unity_tables:
                tables.extend(unity_tables)
            else:
                # Fallback to legacy metastore
                self.logger.info("Unity Catalog not available, trying legacy metastore")
                legacy_tables = self._list_legacy_tables()
                tables.extend(legacy_tables)

            self.logger.info(f"Found {len(tables)} tables")

            return MigrationResult(
                status="success",
                message=f"Successfully listed {len(tables)} tables",
                data={"tables": tables, "count": len(tables)},
            )

        except Exception as e:
            error_msg = f"Error listing tables: {str(e)}"
            self.logger.error(error_msg)
            return MigrationResult(status="failed", message=error_msg, errors=[str(e)])

    def _list_unity_catalog_tables(
        self, catalog: Optional[str] = None, schema: Optional[str] = None
    ) -> List[Dict]:
        """List tables from Unity Catalog across catalogs and schemas.

        If `catalog` or `schema` is provided, restricts to those. Otherwise
        the method enumerates catalogs and schemas and paginates the
        `/unity-catalog/tables` endpoint. Filters to DELTA tables by default.
        """
        tables: List[Dict] = []

        try:
            # Build list of catalogs to query
            if catalog:
                catalogs_to_query = [catalog]
            else:
                catalogs_to_query = [c.get("name") for c in self._list_unity_catalogs()]

            for cat in catalogs_to_query:
                # Build list of schemas for this catalog
                if schema:
                    schemas_to_query = [schema]
                else:
                    schemas_to_query = [
                        s.get("name") for s in self._list_unity_schemas(cat)
                    ]

                for sch in schemas_to_query:
                    page_token = None
                    while True:
                        params = {"catalog_name": cat, "schema_name": sch}
                        if page_token:
                            params["page_token"] = page_token

                        response = self._request(
                            "GET",
                            f"{self.base_url}/unity-catalog/tables",
                            params=params,
                            headers=self.headers,
                        )

                        if not response or response.status_code != 200:
                            if response is None:
                                self.logger.warning(
                                    f"Failed to list tables for {cat}.{sch}: no response"
                                )
                            else:
                                self.logger.warning(
                                    f"Failed to list tables for {cat}.{sch}: {response.status_code}"
                                )
                            break

                        body = response.json()
                        catalog_tables = body.get("tables", [])

                        for table in catalog_tables:
                            fmt = (
                                table.get("data_source_format")
                                or table.get("format")
                                or table.get("storage_format")
                                or ""
                            )
                            if fmt and str(fmt).upper() != "DELTA":
                                continue

                            table_info = {
                                "name": table.get("name"),
                                "catalog": table.get("catalog_name", cat),
                                "schema": table.get("schema_name", sch),
                                "table_type": table.get("table_type", "MANAGED"),
                                "data_source_format": fmt,
                                "storage_location": table.get("storage_location", ""),
                                # Normalize column metadata for downstream UI
                                "columns": self._normalize_columns(
                                    table.get("columns", [])
                                ),
                                "size_bytes": table.get("size_bytes", 0),
                                "created_at": table.get("created_at"),
                                "updated_at": table.get("updated_at"),
                                "owner": table.get("owner", ""),
                                "comment": table.get("comment", ""),
                            }
                            tables.append(table_info)

                        # Databricks uses next_page_token or next_page in responses
                        page_token = body.get("next_page_token") or body.get(
                            "next_page"
                        )
                        if not page_token:
                            break

        except Exception as e:
            self.logger.error(f"Error listing Unity Catalog tables: {e}")

        return tables

    def _list_unity_catalogs(self) -> List[Dict]:
        """Return list of unity catalogs available in the workspace."""
        try:
            response = self._request(
                "GET", f"{self.base_url}/unity-catalog/catalogs", headers=self.headers
            )
            if response and response.status_code == 200:
                return response.json().get("catalogs", [])
            else:
                if response is None:
                    self.logger.debug("Failed to list unity catalogs: no response")
                else:
                    self.logger.debug(
                        f"Failed to list unity catalogs: {response.status_code}"
                    )
        except Exception as e:
            self.logger.debug(f"Error listing unity catalogs: {e}")

        return []

    def _normalize_columns(self, columns: List[Dict]) -> List[Dict]:
        """Normalize column entries to have name and type fields.

        Databricks Unity Catalog may return columns with different shapes.
        This helper ensures a consistent dict: {name, type, nullable}.
        """

        def _render_type(t):
            """Render various type representations into a SQL-like type string."""
            try:
                if t is None:
                    return "string"
                if isinstance(t, str):
                    return t
                if isinstance(t, (int, float, bool)):
                    return str(t)
                # t may be a dict with different shapes depending on API
                if isinstance(t, dict):
                    # decimal type
                    if t.get("type") in ("decimal", "numeric"):
                        p = t.get("precision")
                        s = t.get("scale")
                        if p is not None and s is not None:
                            return f"decimal({p},{s})"
                        return "decimal"

                    # struct-like with fields
                    if "fields" in t and isinstance(t["fields"], list):
                        parts = []
                        for f in t["fields"]:
                            fname = f.get("name") or f.get("field")
                            ftype = _render_type(
                                f.get("type") or f.get("data_type") or f.get("dtype")
                            )
                            parts.append(f"{fname}:{ftype}")
                        return f"struct<{', '.join(parts)}>"

                    # array/list with elementType
                    if "elementType" in t:
                        return f"array<{_render_type(t.get('elementType'))}>"

                    # sometimes type is nested under 'type'
                    nested = t.get("type")
                    if nested is not None and nested is not t:
                        return _render_type(nested)

                    # fallback to JSON string
                    try:
                        return json.dumps(t)
                    except Exception:
                        return str(t)

                # fallback
                return str(t)
            except Exception:
                return "string"

        normalized = []
        for c in columns or []:
            try:
                if not isinstance(c, dict):
                    # If c is a string like 'colname type', try to split
                    parts = str(c).split()
                    name = parts[0]
                    ctype = " ".join(parts[1:]) if len(parts) > 1 else "string"
                    normalized.append({"name": name, "type": ctype, "nullable": True})
                    continue

                # common fields mapping
                name = (
                    c.get("name")
                    or c.get("column_name")
                    or c.get("field")
                    or c.get("col_name")
                )

                # Try multiple locations for the type information
                raw_type = None
                for key in ("type", "data_type", "dtype", "type_name", "dataType"):
                    if key in c and c.get(key) is not None:
                        raw_type = c.get(key)
                        break

                # Some schemas include nested field descriptors under 'schema' or 'type'
                if (
                    raw_type is None
                    and c.get("schema")
                    and isinstance(c.get("schema"), dict)
                ):
                    raw_type = c.get("schema")

                # Render the raw type into a string
                ctype = (
                    _render_type(raw_type)
                    if raw_type is not None
                    else _render_type(c.get("type"))
                )

                nullable = True
                if "nullable" in c:
                    nullable = bool(c.get("nullable"))
                elif "is_nullable" in c:
                    nullable = bool(c.get("is_nullable"))
                elif (
                    isinstance(c.get("type"), dict)
                    and c.get("type").get("nullable") is not None
                ):
                    nullable = bool(c.get("type").get("nullable"))

                normalized.append({"name": name, "type": ctype, "nullable": nullable})
            except Exception:
                # best-effort: skip problematic entries
                continue

        return normalized

    def get_table_columns(
        self, catalog: str, schema: str, table_name: str
    ) -> List[Dict]:
        """Fetch column details for a specific table using Unity Catalog describe endpoint.

        This performs an API call if available; falls back to the cached table list.
        """
        try:
            # Prefer Unity Catalog describe/table endpoint if available
            params = {
                "catalog_name": catalog,
                "schema_name": schema,
                "table_name": table_name,
            }
            response = self.session.get(
                f"{self.base_url}/unity-catalog/tables/get",
                headers=self.headers,
                params=params,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                body = response.json()
                cols = body.get("columns") or body.get("schema", {}).get("fields") or []
                return self._normalize_columns(cols)
        except Exception:
            self.logger.debug(
                f"Failed to get detailed columns for {catalog}.{schema}.{table_name}"
            )

        # Fallback: list tables and find matching entry
        tables_result = self.list_tables(catalog=catalog, schema=schema)
        if tables_result.is_success:
            for t in tables_result.data.get("tables", []):
                if (
                    t.get("name") == table_name
                    and t.get("schema") == schema
                    and t.get("catalog") == catalog
                ):
                    return t.get("columns", [])

        return []

    def _list_unity_schemas(self, catalog_name: str) -> List[Dict]:
        """Return list of schemas for a given unity catalog."""
        try:
            response = self._request(
                "GET",
                f"{self.base_url}/unity-catalog/schemas",
                params={"catalog_name": catalog_name},
                headers=self.headers,
            )
            if response and response.status_code == 200:
                return response.json().get("schemas", [])
            else:
                if response is None:
                    self.logger.debug(
                        f"Failed to list schemas for {catalog_name}: no response"
                    )
                else:
                    self.logger.debug(
                        f"Failed to list schemas for {catalog_name}: {response.status_code}"
                    )
        except Exception as e:
            self.logger.debug(f"Error listing schemas for {catalog_name}: {e}")

        return []

    def _list_legacy_tables(self) -> List[Dict]:
        """List tables from legacy Hive metastore."""
        # For demo purposes, return sample table data
        # In a real implementation, you would query the Hive metastore
        return [
            {
                "name": "customer_data",
                "catalog": "hive_metastore",
                "schema": "default",
                "table_type": "MANAGED",
                "data_source_format": "DELTA",
                "storage_location": "/databricks-datasets/customer_data",
                "columns": [
                    {"name": "customer_id", "type": "bigint", "nullable": False},
                    {"name": "name", "type": "string", "nullable": True},
                    {"name": "email", "type": "string", "nullable": True},
                    {"name": "created_date", "type": "timestamp", "nullable": True},
                ],
                "size_bytes": 5242880,  # 5MB
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-15T12:00:00Z",
                "owner": "admin",
                "comment": "Customer master data",
            },
            {
                "name": "sales_transactions",
                "catalog": "hive_metastore",
                "schema": "sales",
                "table_type": "MANAGED",
                "data_source_format": "DELTA",
                "storage_location": "/databricks-datasets/sales_transactions",
                "columns": [
                    {"name": "transaction_id", "type": "string", "nullable": False},
                    {"name": "customer_id", "type": "bigint", "nullable": False},
                    {"name": "amount", "type": "decimal(10,2)", "nullable": False},
                    {"name": "transaction_date", "type": "date", "nullable": False},
                ],
                "size_bytes": 104857600,  # 100MB
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-20T08:30:00Z",
                "owner": "data_team",
                "comment": "Daily sales transaction data",
            },
        ]

    def get_cluster_info(self) -> MigrationResult:
        """
        Get information about clusters in the workspace.

        Returns:
            MigrationResult containing cluster information
        """
        try:
            self.logger.info("Getting cluster information")

            response = self._request(
                "GET", f"{self.base_url}/clusters/list", headers=self.headers
            )

            if response and response.status_code == 200:
                clusters = response.json().get("clusters", [])
                self.logger.info(f"Found {len(clusters)} clusters")

                return MigrationResult(
                    status="success",
                    message=f"Successfully retrieved {len(clusters)} clusters",
                    data={"clusters": clusters, "count": len(clusters)},
                )
            else:
                error_msg = f"Failed to get cluster info: {response.status_code}"
                self.logger.error(error_msg)
                return MigrationResult(
                    status="failed",
                    message=error_msg,
                    errors=[f"HTTP {response.status_code}: {response.text}"],
                )

        except Exception as e:
            error_msg = f"Error getting cluster info: {str(e)}"
            self.logger.error(error_msg)
            return MigrationResult(status="failed", message=error_msg, errors=[str(e)])

    def get_workspace_info(self) -> MigrationResult:
        """
        Get general workspace information.

        Returns:
            MigrationResult containing workspace information
        """
        try:
            self.logger.info("Getting workspace information")

            # Get current user info
            user_response = self._request(
                "GET", f"{self.base_url}/preview/scim/v2/Me", headers=self.headers
            )

            workspace_info = {
                "workspace_url": self.workspace_url,
                "api_version": self.get_config_value("databricks.api_version", "2.0"),
                "connection_status": "connected",
            }

            if user_response.status_code == 200:
                user_info = user_response.json()
                workspace_info["current_user"] = {
                    "id": user_info.get("id"),
                    "userName": user_info.get("userName"),
                    "displayName": user_info.get("displayName"),
                    "active": user_info.get("active"),
                }

            return MigrationResult(
                status="success",
                message="Successfully retrieved workspace information",
                data=workspace_info,
            )

        except Exception as e:
            error_msg = f"Error getting workspace info: {str(e)}"
            self.logger.error(error_msg)
            return MigrationResult(status="failed", message=error_msg, errors=[str(e)])

    def list_jobs(self) -> MigrationResult:
        """List scheduled jobs (simple schedule jobs only).

        Returns a simplified list with job id, name, and schedule if available.
        """
        try:
            response = self._request(
                "GET", f"{self.base_url}/jobs/list", headers=self.headers
            )

            if response and response.status_code == 200:
                jobs = response.json().get("jobs", [])
                simple_jobs = []
                for job in jobs:
                    # Try to extract simple schedule info
                    job_settings = job.get("settings", {})
                    schedule = job_settings.get("schedule") or job_settings.get(
                        "schedule", {}
                    )
                    # Keep only simple cron or periodic schedule fields
                    schedule_summary = None
                    if isinstance(schedule, dict):
                        # Databricks job schedule may include 'quartz_cron_expression' or 'cron'
                        schedule_summary = schedule.get(
                            "quartz_cron_expression"
                        ) or schedule.get("cron")

                    simple_jobs.append(
                        {
                            "job_id": job.get("job_id"),
                            "name": job.get("settings", {}).get("name")
                            or job.get("name")
                            or f"job_{job.get('job_id')}",
                            "schedule": schedule_summary,
                        }
                    )

                return MigrationResult(
                    status="success",
                    message=f"Listed {len(simple_jobs)} jobs",
                    data={"jobs": simple_jobs, "count": len(simple_jobs)},
                )

            else:
                return MigrationResult(
                    status="failed",
                    message=f"Failed to list jobs: {response.status_code}",
                    errors=[response.text],
                )

        except Exception as e:
            return MigrationResult(status="failed", message=str(e), errors=[str(e)])


if __name__ == "__main__":
    # Test the connector
    import os

    from dotenv import load_dotenv

    from ..utils.base import ConfigManager

    load_dotenv()

    config_manager = ConfigManager()
    config = config_manager.config

    # Override with environment variables for testing
    config.setdefault("databricks", {}).setdefault("workspace", {})
    config["databricks"]["workspace"]["url"] = os.getenv("DATABRICKS_WORKSPACE_URL")
    config["databricks"]["workspace"]["token"] = os.getenv("DATABRICKS_ACCESS_TOKEN")

    if (
        config["databricks"]["workspace"]["url"]
        and config["databricks"]["workspace"]["token"]
    ):
        connector = DatabricksConnector(config)

        # Test connection
        result = connector.test_connection()
        if result.is_success:
            print("✅ Connection successful!")

            # List notebooks
            notebooks_result = connector.list_notebooks()
            if notebooks_result.is_success:
                print(f"Found {notebooks_result.data['count']} notebooks")

            # List tables
            tables_result = connector.list_tables()
            if tables_result.is_success:
                print(f"Found {tables_result.data['count']} tables")
        else:
            print(f"❌ Connection failed: {result.message}")
    else:
        print(
            "Please set DATABRICKS_WORKSPACE_URL and DATABRICKS_ACCESS_TOKEN environment variables"
        )
