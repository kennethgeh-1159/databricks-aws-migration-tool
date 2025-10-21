"""TCO estimation helpers for Databricks -> AWS migration.

This module provides approximate cost estimates using available cluster
metadata and table sizes. All numbers are estimates and should be validated
with real billing/usage data for production decisions.
"""

import datetime
import math
from typing import Any, Dict, List, Optional, Tuple

from utils.base import ConfigManager

DEFAULT_DBU_PRICE_PER_HOUR = 0.55  # fallback USD per DBU-hour
DEFAULT_S3_PRICE_PER_GB_MONTH = 0.023
BUFFER_PERCENT = 0.15

# Default EMR serverless vCPU/memory rates (fallback)
DEFAULT_EMR_VCPU_HOUR = 0.052624
DEFAULT_EMR_MEMORY_GB_HOUR = 0.0057785


def _hours_between(start_ts: int, end_ts: int) -> float:
    # timestamps in ms or seconds; normalize
    if start_ts > 1e12:  # ms
        start = datetime.datetime.fromtimestamp(start_ts / 1000.0)
    else:
        start = datetime.datetime.fromtimestamp(start_ts)
    if end_ts > 1e12:
        end = datetime.datetime.fromtimestamp(end_ts / 1000.0)
    else:
        end = datetime.datetime.fromtimestamp(end_ts)
    diff = end - start
    return diff.total_seconds() / 3600.0


def estimate_databricks_costs_from_clusters(
    clusters: List[Dict[str, Any]],
    lookback_days: int = 30,
    dbu_price_per_hour: Optional[float] = None,
    utilization_factor: float = 0.25,
    config: Optional[ConfigManager] = None,
) -> Dict[str, Any]:
    """Estimate Databricks compute costs from cluster metadata.

    clusters: list of cluster dicts from Databricks API /clusters/list
    This function attempts to estimate hours used in the last lookback_days.
    If cluster start/terminate timestamps are available, uses them; otherwise
    uses a utilization factor to estimate active hours.
    """
    now = datetime.datetime.utcnow()
    window_start = now - datetime.timedelta(days=lookback_days)

    total_dbu_hours = 0.0
    per_cluster = []

    # Determine DBU price from config if not explicitly provided
    if dbu_price_per_hour is None:
        try:
            dbu_price_per_hour = (
                float(
                    config.get("costs.databricks.dbu_hour", DEFAULT_DBU_PRICE_PER_HOUR)
                )
                if config
                else DEFAULT_DBU_PRICE_PER_HOUR
            )
        except Exception:
            dbu_price_per_hour = DEFAULT_DBU_PRICE_PER_HOUR

    for c in clusters:
        cluster_id = c.get("cluster_id") or c.get("cluster_id")
        num_workers = (
            c.get("num_workers") or c.get("autoscale", {}).get("min_workers") or 0
        )
        driver_type = c.get("driver_node_type_id") or c.get("node_type_id")
        worker_type = c.get("node_type_id")

        # Estimate vCPU count per node as a proxy (very rough)
        def _vcpus_from_node(node_type: str) -> int:
            if not node_type:
                return 4
            if "xlarge" in node_type:
                return 4
            if "2xlarge" in node_type or "large" in node_type:
                return 8
            return 4

        # Estimate DBU per hour: proxy 1 DBU per vCPU
        driver_vcpus = _vcpus_from_node(driver_type)
        worker_vcpus = _vcpus_from_node(worker_type)

        # Estimate hours in lookback window
        hours = 24 * lookback_days * utilization_factor

        # If cluster has 'start_time' and 'terminated_time' estimate accurately
        start_ts = c.get("start_time") or c.get("created_time")
        end_ts = c.get("terminated_time") or c.get("end_time")
        if start_ts:
            try:
                if not end_ts:
                    # still running: use now
                    end_dt = now
                    start_dt = datetime.datetime.fromtimestamp(
                        start_ts / (1000.0 if start_ts > 1e12 else 1.0)
                    )
                    overlap_start = max(start_dt, window_start)
                    hours = (now - overlap_start).total_seconds() / 3600.0
                else:
                    hours = _hours_between(
                        max(start_ts, int(window_start.timestamp())), end_ts
                    )
            except Exception:
                pass

        # Compute DBU-hours: approximate driver + workers
        workers = max(int(num_workers or 0), 0)
        dbu_per_hour = driver_vcpus + workers * worker_vcpus
        dbu_hours = dbu_per_hour * max(hours, 0)

        cost = dbu_hours * dbu_price_per_hour

        per_cluster.append(
            {
                "cluster_id": cluster_id,
                "driver_type": driver_type,
                "worker_type": worker_type,
                "workers": workers,
                "dbu_hours": dbu_hours,
                "estimated_cost_usd": cost,
            }
        )

        total_dbu_hours += dbu_hours

    total_compute_cost = total_dbu_hours * dbu_price_per_hour

    return {
        "total_dbu_hours": total_dbu_hours,
        "compute_cost_usd": total_compute_cost,
        "per_cluster": per_cluster,
    }


def estimate_storage_costs_from_tables(
    tables: List[Dict[str, Any]], config: Optional[ConfigManager] = None
) -> Dict[str, Any]:
    total_bytes = sum([t.get("size_bytes", 0) for t in tables])
    total_gb = total_bytes / (1024**3)
    try:
        s3_price = (
            float(
                config.get(
                    "costs.aws.s3_standard_gb_month", DEFAULT_S3_PRICE_PER_GB_MONTH
                )
            )
            if config
            else DEFAULT_S3_PRICE_PER_GB_MONTH
        )
    except Exception:
        s3_price = DEFAULT_S3_PRICE_PER_GB_MONTH

    monthly_storage_cost = total_gb * s3_price
    return {
        "total_bytes": total_bytes,
        "total_gb": total_gb,
        "monthly_storage_cost_usd": monthly_storage_cost,
    }


def map_to_aws_emr_estimate(
    databricks_compute: Dict[str, Any], config: Optional[ConfigManager] = None
) -> Dict[str, Any]:
    """Map Databricks compute estimate to EMR approximate costs.

    This uses coarse mappings and on-demand hourly prices per instance.
    """
    total_dbu_hours = databricks_compute.get("total_dbu_hours", 0)
    # Convert DBU-hours to equivalent vCPU-hours
    vcpu_hours = total_dbu_hours

    # Determine vCPU and memory pricing from config (EMR serverless style)
    try:
        vcpu_price = (
            float(
                config.get("costs.aws.emr_serverless_vcpu_hour", DEFAULT_EMR_VCPU_HOUR)
            )
            if config
            else DEFAULT_EMR_VCPU_HOUR
        )
    except Exception:
        vcpu_price = DEFAULT_EMR_VCPU_HOUR

    try:
        mem_price = (
            float(
                config.get(
                    "costs.aws.emr_serverless_memory_gb_hour",
                    DEFAULT_EMR_MEMORY_GB_HOUR,
                )
            )
            if config
            else DEFAULT_EMR_MEMORY_GB_HOUR
        )
    except Exception:
        mem_price = DEFAULT_EMR_MEMORY_GB_HOUR

    # Estimate memory as 2GB per vCPU as a rough heuristic
    memory_gb_hours = vcpu_hours * 2

    compute_cost = vcpu_hours * vcpu_price + memory_gb_hours * mem_price

    return {
        "model": "emr_serverless_estimate",
        "vcpu_hours": vcpu_hours,
        "memory_gb_hours": memory_gb_hours,
        "compute_cost_usd": compute_cost,
    }


def add_buffer(baseline_usd: float, percent: float = BUFFER_PERCENT) -> float:
    return baseline_usd * (1.0 + percent)


def generate_simple_roi(
    databricks_monthly: float, aws_monthly: float
) -> Dict[str, Any]:
    """Return monthly/annual and ROI timeline estimate.

    ROI timeline estimated as months to recover migration cost; we assume a
    migration one-time cost placeholder of 10x monthly savings.
    """
    monthly_savings = databricks_monthly - aws_monthly
    annual_savings = monthly_savings * 12
    migration_cost_placeholder = max(1.0, monthly_savings * 10)
    months_to_recover = (
        migration_cost_placeholder / monthly_savings if monthly_savings > 0 else None
    )

    return {
        "monthly_savings_usd": monthly_savings,
        "annual_savings_usd": annual_savings,
        "migration_cost_estimate_usd": migration_cost_placeholder,
        "months_to_recover": months_to_recover,
    }


def bedrock_recommendations_placeholder(summary: Dict[str, Any]) -> List[str]:
    # Placeholder: if AWS Bedrock access is configured, this could call a model
    # to suggest rightsizing or instance type adjustments. For now return simple tips.
    tips = []
    # Identify clusters with high DBU-hours per worker
    for c in summary.get("per_cluster", [])[:5]:
        if c.get("dbu_hours", 0) > 1000:
            tips.append(
                f"Cluster {c.get('cluster_id')} has high usage; consider profiling and right-sizing."
            )

    tips.append(
        "Consider using spot instances or EMR Managed Scaling for cost savings."
    )
    tips.append("Archive infrequently accessed data to Glacier/IA to reduce S3 costs.")
    return tips
