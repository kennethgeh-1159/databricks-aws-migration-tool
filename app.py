"""
Main Streamlit application for Databricks AWS Migration Tool.
Entry point for the web interface.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st
from dotenv import load_dotenv

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from connectors.databricks_connector import DatabricksConnector

# Import our modules
from utils.base import ConfigManager, LoggerSetup

# Load environment variables
load_dotenv()

# Initialize configuration
config_manager = ConfigManager()
config = config_manager.config

# Setup logging
logger = LoggerSetup.setup_logging(config)


def get_configuration() -> Dict[str, Any]:
    """Return a dictionary of app configuration values for the sidebar.

    Values come from environment variables, the loaded config_manager, or sensible defaults.
    """

    app_config: Dict[str, Any] = {}

    # Databricks
    app_config["databricks_workspace_url"] = os.getenv(
        "DATABRICKS_WORKSPACE_URL",
        config_manager.config.get("databricks", {}).get("workspace", {}).get("url", ""),
    )
    app_config["databricks_access_token"] = os.getenv(
        "DATABRICKS_ACCESS_TOKEN",
        config_manager.config.get("databricks", {})
        .get("workspace", {})
        .get("token", ""),
    )

    # AWS
    app_config["aws_region"] = os.getenv(
        "AWS_REGION", config_manager.config.get("aws", {}).get("region", "us-east-1")
    )
    app_config["s3_bucket"] = os.getenv(
        "S3_BUCKET",
        config_manager.config.get("aws", {}).get("s3", {}).get("bucket", ""),
    )

    # UI defaults
    app_config["max_notebooks"] = int(os.getenv("MAX_NOTEBOOKS_TO_ANALYZE", 3))
    app_config["max_tables"] = int(os.getenv("MAX_TABLES_TO_MIGRATE", 1))

    # Bedrock model default (can be overridden in the Advanced Settings)
    app_config["bedrock_model"] = (
        config_manager.config.get("aws", {})
        .get("bedrock", {})
        .get("model_id", "anthropic.claude-3-sonnet-20240229-v1:0")
    )

    # Debug / flags
    app_config["enable_debug"] = os.getenv("DEBUG", "false").lower() == "true"
    app_config["parallel_processing"] = True

    return app_config


def compute_confidence_score(complexity, issues_count, size_kb):
    """Compute a confidence score (0-100) and label from complexity, issues count and size.

    Heuristics:
    - complexity: simple -> higher base score, moderate -> medium, complex -> low
    - issues_count: each issue reduces confidence (~15% each) up to a cap
    - size_kb: larger notebooks reduce confidence via a size factor

    Returns (confidence_pct:int, label:str)
    """
    try:
        complexity_key = (complexity or "").lower()
    except Exception:
        complexity_key = ""

    complexity_map = {"simple": 1.0, "moderate": 0.6, "complex": 0.3}
    complexity_score = complexity_map.get(complexity_key, 0.5)

    try:
        issues_count = int(issues_count or 0)
    except Exception:
        issues_count = 0

    # each issue reduces confidence by ~15%, clamp at 90%
    issues_penalty = min(issues_count * 0.15, 0.9)

    try:
        size_kb = float(size_kb or 0)
    except Exception:
        size_kb = 0.0

    if size_kb <= 100:
        size_factor = 1.0
    elif size_kb <= 500:
        size_factor = 0.9
    elif size_kb <= 2000:
        size_factor = 0.7
    else:
        size_factor = 0.5

    combined = complexity_score * (1 - issues_penalty) * size_factor
    combined = max(0.0, min(1.0, combined))
    pct = int(round(combined * 100))

    if pct >= 80:
        label = "High"
    elif pct >= 50:
        label = "Medium"
    else:
        label = "Low"

    return pct, label


def main():
    """Main Streamlit application entry point."""

    # Page configuration
    st.set_page_config(
        page_title="Databricks to AWS Migration Tool",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/yourusername/databricks-aws-migration-tool",
            "Report a bug": "https://github.com/yourusername/databricks-aws-migration-tool/issues",
            "About": """
            # Databricks to AWS Migration Tool

            AI-powered tool for migrating Databricks workloads to AWS native services.

            **Features:**
            - 🧠 AI-powered complexity analysis
            - 💰 TCO calculation and optimization
            - 🔄 Automated code conversion
            - 📊 Data migration to S3
            - 📈 Interactive dashboards
            """,
        },
    )

    # --- Page navigation ---
    page = st.sidebar.selectbox(
        "Pages",
        ["Home", "Analysis Results", "TCO Report", "Migration Status"],
        index=0,
    )

    # Custom CSS
    st.markdown(
        """
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .success-banner {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .warning-banner {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    .error-banner {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f1f3f4;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Main header
    st.markdown(
        '<h1 class="main-header">🚀 Databricks to AWS Migration Tool</h1>',
        unsafe_allow_html=True,
    )

    # Introduction
    st.markdown(
        """
    <div class="feature-card">
    <h3>🎯 Transform Your Data Platform</h3>
    <p>Migrate your Databricks workloads to AWS native services with confidence using AI-powered analysis and automated conversion tools.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Feature overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        **🧠 AI Analysis**
        - Complexity scoring
        - Risk assessment
        - Migration recommendations
        """
        )

    with col2:
        st.markdown(
            """
        **💰 Cost Optimization**
        - TCO comparison
        - Savings projection
        - ROI analysis
        """
        )

    with col3:
        st.markdown(
            """
        **🔄 Code Conversion**
        - Template-based conversion
        - EMR compatibility
        - Confidence scoring
        """
        )

    with col4:
        st.markdown(
            """
        **📊 Data Migration**
        - S3 migration
        - Format optimization
        - Validation checks
        """
        )

    # Check if this is the first run
    if "app_initialized" not in st.session_state:
        st.session_state.app_initialized = True
        st.info(
            "👋 Welcome! Please configure your settings in the sidebar to get started."
        )

    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        app_config = get_configuration()

        # --- Dropdowns: Databricks workspace, Bedrock model, S3 bucket ---
        st.markdown("---")
        st.subheader("🧩 Quick Configuration")

        # Databricks workspace selector: look for configured entries
        configured_wss = []
        try:
            db_cfg = config_manager.config.get("databricks", {})
            # if multiple workspaces provided under 'workspaces' key
            if isinstance(db_cfg, dict):
                ws_map = db_cfg.get("workspaces") or {}
                if isinstance(ws_map, dict):
                    configured_wss = [
                        v.get("url") for k, v in ws_map.items() if v.get("url")
                    ]
        except Exception:
            configured_wss = []

        # include current env/config value
        current_ws = app_config.get("databricks_workspace_url")
        if current_ws and current_ws not in configured_wss:
            configured_wss.insert(0, current_ws)

        configured_wss = [w for w in configured_wss if w]
        configured_wss_options = configured_wss + ["Custom..."]
        selected_ws = st.selectbox(
            "Databricks Workspace",
            options=(
                configured_wss_options if configured_wss_options else ["Custom..."]
            ),
            index=0,
        )
        if selected_ws == "Custom...":
            custom_ws = st.text_input(
                "Databricks Workspace URL (no https://)", value=current_ws or ""
            )
            app_config["databricks_workspace_url"] = custom_ws
        else:
            app_config["databricks_workspace_url"] = selected_ws

        # Bedrock model selector
        bedrock_defaults = [
            app_config.get("bedrock_model")
            or "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "ai21.j2-large-instruct",
        ]
        bedrock_choices = list(dict.fromkeys(bedrock_defaults))  # unique
        selected_model = st.selectbox("Bedrock Model", options=bedrock_choices, index=0)
        app_config["bedrock_model"] = selected_model

        # S3 bucket selector: use configured bucket or let user type
        configured_bucket = app_config.get("s3_bucket") or config_manager.config.get(
            "aws", {}
        ).get("s3", {}).get("bucket")
        bucket_input = st.text_input(
            "S3 Bucket for migration artifacts", value=configured_bucket or ""
        )
        app_config["s3_bucket"] = bucket_input

        st.markdown("---")

        # Connection testing section
        st.markdown("---")
        st.subheader("🔍 Connection Tests")

        if st.button("🔗 Test All Connections", type="secondary", width="stretch"):
            test_all_connections(app_config)

        # Quick actions
        st.markdown("---")
        st.subheader("⚡ Quick Actions")

        if st.button("📋 View Sample Data", width="stretch"):
            show_sample_data()

        if st.button("📖 View Documentation", width="stretch"):
            show_documentation()

    # Main content area
    if not validate_configuration(app_config):
        show_configuration_help()
        return

    # --- Page routing ---
    if page == "Home":
        st.header("Home")
        st.markdown("Welcome to the Databricks to AWS Migration Tool dashboard.")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Per-run Bedrock toggle (overrides config_manager for this run)
            try:
                default_bedrock = bool(
                    config_manager.get("features", {}).get(
                        "enable_bedrock_analysis", True
                    )
                )
            except Exception:
                default_bedrock = True

            enable_bedrock_run = st.checkbox(
                "Enable Bedrock analysis for this run",
                value=default_bedrock,
                help="Toggle calling Amazon Bedrock for enriched analysis (requires AWS credentials).",
                key="enable_bedrock_run",
            )

            # Show a warning if user enables Bedrock but AWS credentials aren't available
            if enable_bedrock_run:
                try:
                    import boto3

                    if boto3.Session().get_credentials() is None:
                        st.warning(
                            "Bedrock enabled but no AWS credentials found in the environment. Analysis will attempt Bedrock and may fail."
                        )
                except Exception:
                    st.warning(
                        "Bedrock enabled but boto3 is not available in the environment. Install boto3 in the venv to enable Bedrock calls."
                    )

            if st.button("Run Analysis", key="run_analysis"):
                st.session_state["action"] = "run_analysis"
                # Trigger the analysis flow (placeholder)
                try:
                    # Override the in-memory config flag for this run based on the checkbox
                    try:
                        # Ensure features dict exists
                        cfg = config_manager.config
                        cfg.setdefault("features", {})
                        cfg["features"]["enable_bedrock_analysis"] = bool(
                            enable_bedrock_run
                        )
                    except Exception:
                        pass

                    migration_tool = initialize_migration_tool(app_config)
                    run_full_analysis(migration_tool)
                except Exception as e:
                    st.error(f"Failed to start analysis: {e}")

        with col2:
            if st.button("Generate TCO", key="generate_tco"):
                st.session_state["action"] = "generate_tco"
                # Placeholder: compute/generate TCO
                st.info("TCO generation started (placeholder)")

        with col3:
            if st.button("Start Migration", key="start_migration"):
                st.session_state["action"] = "start_migration"
                # Placeholder: trigger migration workflow
                st.info("Migration workflow started (placeholder)")

        st.markdown("---")
        st.markdown(
            "Use the sidebar to navigate to other pages: Analysis Results, TCO Report, or Migration Status."
        )

    elif page == "Analysis Results":
        st.header("Analysis Results")
        if (
            "migration_results" in st.session_state
            and st.session_state["migration_results"]
        ):
            display_results_dashboard()
        else:
            st.warning("No analysis results available. Run an analysis from Home page.")

    elif page == "TCO Report":
        st.header("TCO Report")
        # Placeholder TCO display
        if st.session_state.get("action") == "generate_tco":
            st.success("TCO report generated (placeholder)")
            st.line_chart({"Cost": [1000, 800, 600, 400]})
        else:
            st.info("Click 'Generate TCO' on the Home page to create a report.")
            import pandas as pd

            tables = results.get("tables", {}).get("tables", [])
            clusters = results.get("clusters", {}).get("clusters", [])

            if not tables and not clusters:
                st.info(
                    "No table or cluster data available to compute TCO. Run analysis first."
                )
                return

            # Estimate Databricks compute costs from clusters (last 30 days)
            from tco.tco import (
                add_buffer,
                bedrock_recommendations_placeholder,
                estimate_databricks_costs_from_clusters,
                estimate_storage_costs_from_tables,
                generate_simple_roi,
                map_to_aws_emr_estimate,
            )

            db_compute = estimate_databricks_costs_from_clusters(
                clusters, lookback_days=30, config=config_manager
            )

            storage = estimate_storage_costs_from_tables(tables, config=config_manager)

            # Map to AWS EMR cost estimate
            aws_emr = map_to_aws_emr_estimate(db_compute, config=config_manager)

            # Add buffer
            db_compute_buffered = add_buffer(db_compute.get("compute_cost_usd", 0))
            db_storage_buffered = add_buffer(storage.get("monthly_storage_cost_usd", 0))

            aws_compute_buffered = add_buffer(aws_emr.get("compute_cost_usd", 0))
            aws_storage_buffered = add_buffer(
                storage.get("monthly_storage_cost_usd", 0)
            )

            # Present results
            comp = {
                "Metric": ["Compute (monthly)", "Storage (monthly)", "Total Monthly"],
                "Databricks (USD)": [
                    f"${db_compute_buffered:,.2f}",
                    f"${db_storage_buffered:,.2f}",
                    f"${(db_compute_buffered + db_storage_buffered):,.2f}",
                ],
                "AWS (USD)": [
                    f"${aws_compute_buffered:,.2f}",
                    f"${aws_storage_buffered:,.2f}",
                    f"${(aws_compute_buffered + aws_storage_buffered):,.2f}",
                ],
            }

            comp_df = pd.DataFrame(comp)
            st.markdown("#### Side-by-side cost comparison (with 15% buffer)")
            st.dataframe(comp_df, width="stretch")

            # Projections and ROI
            db_monthly = db_compute_buffered + db_storage_buffered
            aws_monthly = aws_compute_buffered + aws_storage_buffered
            roi = generate_simple_roi(db_monthly, aws_monthly)

            st.markdown("#### Projections & ROI")
            st.markdown(f"- Databricks monthly (buffered): **${db_monthly:,.2f}**")
            st.markdown(f"- AWS monthly (buffered): **${aws_monthly:,.2f}**")
            st.markdown(
                f"- Estimated monthly savings: **${roi['monthly_savings_usd']:,.2f}**"
            )
            st.markdown(
                f"- Estimated annual savings: **${roi['annual_savings_usd']:,.2f}**"
            )
            if roi.get("months_to_recover"):
                st.markdown(
                    f"- Estimated months to recover migration cost: **{roi['months_to_recover']:.1f} months**"
                )

            # AI-based cost optimization suggestions (Bedrock) — best-effort and guarded
            st.markdown("#### AI Recommendations: Cost Optimization")
            try:
                tips = bedrock_recommendations_placeholder(db_compute)
                for t in tips:
                    st.info(t)
            except Exception as e:
                logger.error(f"Bedrock recommendation error: {e}", exc_info=True)

    app_config["max_notebooks"] = st.slider(
        "Max Notebooks to Analyze",
        min_value=1,
        max_value=20,
        value=int(os.getenv("MAX_NOTEBOOKS_TO_ANALYZE", 3)),
        help="Number of notebooks to analyze",
        key="max_notebooks",
    )

    app_config["max_tables"] = st.slider(
        "Max Tables to Migrate",
        min_value=1,
        max_value=10,
        value=int(os.getenv("MAX_TABLES_TO_MIGRATE", 1)),
        help="Number of tables to migrate in demo",
        key="max_tables",
    )

    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        app_config["bedrock_model"] = st.selectbox(
            "Bedrock Model",
            options=[
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0",
            ],
            index=0,
            help="AI model for code analysis",
            key="bedrock_model",
        )

        app_config["enable_debug"] = st.checkbox(
            "Enable Debug Logging",
            value=os.getenv("DEBUG", "false").lower() == "true",
            help="Enable detailed logging for troubleshooting",
            key="debug_mode",
        )

        app_config["parallel_processing"] = st.checkbox(
            "Enable Parallel Processing",
            value=True,
            help="Process multiple items in parallel",
            key="parallel_processing",
        )

    return app_config


def validate_configuration(app_config: Dict[str, Any]) -> bool:
    """Validate that required configuration is provided."""

    required_fields = [
        ("databricks_workspace_url", "Databricks Workspace URL"),
        ("databricks_access_token", "Databricks Access Token"),
        ("s3_bucket", "S3 Bucket Name"),
    ]

    missing_fields = []
    for field, display_name in required_fields:
        if not app_config.get(field):
            missing_fields.append(display_name)

    if missing_fields:
        st.sidebar.error(f"❌ Missing: {', '.join(missing_fields)}")
        return False

    st.sidebar.success("✅ Configuration Complete")
    return True


def show_configuration_help():
    """Show configuration help when required fields are missing."""

    st.markdown(
        """
    <div class="warning-banner">
    <h3>⚠️ Configuration Required</h3>
    <p>Please complete the configuration in the sidebar before proceeding.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    ## 📋 Setup Instructions

    ### 1. 📊 Databricks Configuration

    **Workspace URL**: Your Databricks workspace URL
    - Example: `your-workspace.cloud.databricks.com`
    - Don't include `https://`

    **Access Token**: Generate a personal access token
    1. Go to Databricks workspace
    2. Click your profile → User Settings
    3. Go to Access Tokens tab
    4. Click "Generate New Token"
    5. Copy the token value

    ### 2. ☁️ AWS Configuration

    **AWS Region**: Select your preferred AWS region
    - Choose the region closest to your data
    - Ensure Bedrock is available in the selected region

    **S3 Bucket**: Create an S3 bucket for migration data
    ```bash
    aws s3 mb s3://your-migration-bucket
    ```

    ### 3. 🔑 AWS Credentials

    Ensure AWS credentials are configured:

    **Option 1: AWS CLI**
    ```bash
    aws configure
    ```

    **Option 2: Environment Variables**
    ```bash
    export AWS_ACCESS_KEY_ID=your-access-key
    export AWS_SECRET_ACCESS_KEY=your-secret-key
    ```

    **Option 3: IAM Roles** (recommended for production)
    - Use IAM roles when running on AWS infrastructure

    ### 4. 🧠 Amazon Bedrock Setup

    1. Go to AWS Console → Amazon Bedrock
    2. Navigate to "Model access"
    3. Enable access to Anthropic Claude models
    4. Wait for approval (usually instant)
    """
    )


def test_all_connections(app_config: Dict[str, Any]):
    """Test connections to all external services."""

    if not validate_configuration(app_config):
        st.sidebar.error("❌ Please complete configuration first")
        return

    with st.sidebar:
        with st.spinner("🔍 Testing connections..."):
            results = {}

            # Test Databricks
            try:
                # Update global config with sidebar values
                test_config = config.copy()
                test_config.setdefault("databricks", {}).setdefault("workspace", {})
                test_config["databricks"]["workspace"]["url"] = app_config[
                    "databricks_workspace_url"
                ]
                test_config["databricks"]["workspace"]["token"] = app_config[
                    "databricks_access_token"
                ]

                db_connector = DatabricksConnector(test_config)
                db_result = db_connector.test_connection()
                results["databricks"] = db_result.is_success

                if not db_result.is_success:
                    st.error(f"Databricks: {db_result.message}")

            except Exception as e:
                results["databricks"] = False
                st.error(f"Databricks error: {str(e)}")

            # Test Bedrock (placeholder - we'll implement this later)
            try:
                import boto3
                from botocore.exceptions import (
                    NoCredentialsError,
                    PartialCredentialsError,
                )

                # Use session to check for credentials first to give a clearer message
                session = boto3.Session()
                creds = session.get_credentials()
                if not creds:
                    raise NoCredentialsError()

                bedrock_client = session.client(
                    "bedrock", region_name=app_config["aws_region"]
                )
                bedrock_client.list_foundation_models()
                results["bedrock"] = True
            except NoCredentialsError:
                results["bedrock"] = False
                st.warning(
                    "Bedrock: AWS credentials not found. Set environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) or configure an IAM role/EC2 instance profile. You can also run `aws configure` to set up credentials locally."
                )
            except PartialCredentialsError:
                results["bedrock"] = False
                st.warning(
                    "Bedrock: Incomplete AWS credentials detected (partial). Check your AWS environment variables or ~/.aws/credentials file."
                )
            except Exception as e:
                results["bedrock"] = False
                st.error(f"Bedrock error: {str(e)}")

            # Test S3
            try:
                import boto3
                from botocore.exceptions import (
                    ClientError,
                    NoCredentialsError,
                    PartialCredentialsError,
                )

                session = boto3.Session()
                creds = session.get_credentials()
                if not creds:
                    raise NoCredentialsError()

                s3_client = session.client("s3", region_name=app_config["aws_region"])
                s3_client.head_bucket(Bucket=app_config["s3_bucket"])
                results["s3"] = True
            except NoCredentialsError:
                results["s3"] = False
                st.warning(
                    "S3: AWS credentials not found. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or configure credentials with `aws configure`."
                )
            except PartialCredentialsError:
                results["s3"] = False
                st.warning(
                    "S3: Incomplete AWS credentials (partial). Check your environment variables or AWS credentials file."
                )
            except ClientError as e:
                results["s3"] = False
                # Show a simpler message for common client errors (like 404 / access denied)
                st.error(
                    f"S3 error: {e.response.get('Error', {}).get('Message', str(e))}"
                )
            except Exception as e:
                results["s3"] = False
                st.error(f"S3 error: {str(e)}")

        # Display results
        st.markdown("#### 🔍 Connection Status")

        for service, status in results.items():
            if status:
                st.success(f"✅ {service.title()}")
            else:
                st.error(f"❌ {service.title()}")

        # Overall status
        if all(results.values()):
            st.success("🎉 All connections successful!")
        else:
            st.warning("⚠️ Some connections failed. Check configuration.")


def initialize_migration_tool(app_config: Dict[str, Any]):
    """Initialize the migration tool with configuration."""

    # Update global config with sidebar values
    updated_config = config.copy()
    updated_config.setdefault("databricks", {}).setdefault("workspace", {})
    updated_config["databricks"]["workspace"]["url"] = app_config[
        "databricks_workspace_url"
    ]
    updated_config["databricks"]["workspace"]["token"] = app_config[
        "databricks_access_token"
    ]
    updated_config["aws"]["region"] = app_config["aws_region"]
    updated_config["aws"]["s3"]["bucket"] = app_config["s3_bucket"]

    # For now, return a simple object - we'll expand this later
    class MigrationTool:
        def __init__(self, config):
            self.config = config
            self.databricks_connector = DatabricksConnector(config)

    return MigrationTool(updated_config)


def run_full_analysis(migration_tool):
    """Run comprehensive migration analysis."""

    try:
        # Initialize result variables to avoid UnboundLocalError if an API call raises
        notebooks_result = None
        tables_result = None
        jobs_result = None
        clusters_result = None
        notebook_analyses = []

        with st.spinner("🚀 Running comprehensive migration analysis..."):
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Test connection
            status_text.text("Step 1/5: Testing Databricks connection...")
            progress_bar.progress(0.2)

            connection_result = migration_tool.databricks_connector.test_connection()
            if not connection_result.is_success:
                st.error(f"❌ Connection failed: {connection_result.message}")
                return

            # Step 2: List notebooks
            status_text.text("Step 2/7: Discovering notebooks (full workspace scan)...")
            progress_bar.progress(0.3)

            # Full workspace export of notebooks (no limit) — this matches previous behavior
            notebooks_result = migration_tool.databricks_connector.list_notebooks()
            if not notebooks_result or not getattr(
                notebooks_result, "is_success", False
            ):
                msg = getattr(notebooks_result, "message", "Failed to list notebooks")
                st.error(f"❌ Failed to list notebooks: {msg}")
                return

            # Step 3: List tables
            status_text.text("Step 3/7: Discovering tables...")
            progress_bar.progress(0.45)

            tables_result = migration_tool.databricks_connector.list_tables()
            if not tables_result or not getattr(tables_result, "is_success", False):
                msg = getattr(tables_result, "message", "Failed to list tables")
                st.error(f"❌ Failed to list tables: {msg}")
                return

            # Step 4: List scheduled jobs
            status_text.text("Step 4/8: Listing scheduled jobs...")
            progress_bar.progress(0.55)

            jobs_result = migration_tool.databricks_connector.list_jobs()
            if not jobs_result or not getattr(jobs_result, "is_success", False):
                msg = getattr(jobs_result, "message", "Failed to list jobs")
                st.warning(f"⚠️ Failed to list jobs: {msg}")
                jobs_data = {"jobs": [], "count": 0}
            else:
                jobs_data = jobs_result.data

            # Step 5: Collect cluster information (for usage / TCO)
            status_text.text("Step 5/8: Collecting cluster info...")
            progress_bar.progress(0.65)
            clusters_result = migration_tool.databricks_connector.get_cluster_info()
            if not clusters_result or not getattr(clusters_result, "is_success", False):
                msg = getattr(clusters_result, "message", "Failed to get cluster info")
                st.warning(f"⚠️ Failed to get cluster info: {msg}")
                clusters_data = {"clusters": [], "count": 0}
            else:
                clusters_data = clusters_result.data

            # Step 6: Analyze notebooks
            status_text.text("Step 6/8: Analyzing notebooks...")
            progress_bar.progress(0.75)

            from analyzers.notebook_analyzer import analyze_notebooks_batch

            notebooks = (
                notebooks_result.data.get("notebooks", [])
                if notebooks_result and getattr(notebooks_result, "data", None)
                else []
            )
            # Pass the global config_manager so analyzer can enable Bedrock and use config
            notebook_analyses = analyze_notebooks_batch(
                notebooks, config=config_manager
            )

            # Step 6: Generate results
            status_text.text("Step 6/7: Generating results and inventory...")
            progress_bar.progress(0.9)

            # Determine a timestamp (use pandas if available for consistency, otherwise fall back)
            try:
                import pandas as pd

                now = str(pd.Timestamp.now())
            except Exception:
                from datetime import datetime

                now = datetime.utcnow().isoformat()

            inventory = {
                "notebooks": notebooks,
                "notebook_analyses": notebook_analyses,
                "tables": (
                    tables_result.data.get("tables", [])
                    if tables_result and getattr(tables_result, "data", None)
                    else []
                ),
                "jobs": jobs_data.get("jobs", []),
                "clusters": clusters_data.get("clusters", []),
                "generated_at": now,
            }

            # Save inventory JSON
            import json
            import os

            os.makedirs("data/results", exist_ok=True)
            with open("data/results/inventory.json", "w") as f:
                json.dump(inventory, f, indent=2)

            # Step 7: finalize
            status_text.text("Step 7/7: Finalizing...")
            progress_bar.progress(1.0)

            # Store results in session state
            st.session_state.migration_results = {
                "notebooks": (
                    notebooks_result.data
                    if notebooks_result and getattr(notebooks_result, "data", None)
                    else {}
                ),
                "notebook_analyses": notebook_analyses,
                "tables": (
                    tables_result.data
                    if tables_result and getattr(tables_result, "data", None)
                    else {}
                ),
                "jobs": jobs_data,
                "clusters": clusters_data,
                "analysis_completed": True,
                "timestamp": now,
            }

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Show success message
        st.success("✅ Migration analysis completed successfully!")

        # Display summary
        display_analysis_summary()

    except Exception as e:
        st.error(f"❌ Error during analysis: {e}")
        logger.error(f"Analysis error: {e}", exc_info=True)


def run_quick_analysis(migration_tool):
    """Run quick notebook analysis only."""

    try:
        with st.spinner("🔍 Running quick notebook analysis..."):
            notebooks_result = migration_tool.databricks_connector.list_notebooks()

            if notebooks_result.is_success:
                st.session_state.migration_results = {
                    "notebooks": notebooks_result.data,
                    "analysis_type": "quick",
                    "timestamp": str(pd.Timestamp.now()),
                }
                st.success("✅ Quick analysis completed!")
                display_analysis_summary()
            else:
                st.error(f"❌ Analysis failed: {notebooks_result.message}")

    except Exception as e:
        st.error(f"❌ Error during quick analysis: {e}")
        logger.error(f"Quick analysis error: {e}", exc_info=True)


def display_analysis_summary():
    """Display analysis summary metrics."""

    if "migration_results" not in st.session_state:
        return

    results = st.session_state.migration_results

    st.markdown("### 📊 Analysis Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        notebook_count = results.get("notebooks", {}).get("count", 0)
        st.metric(
            "📓 Notebooks Found", notebook_count, help="Number of notebooks discovered"
        )

    with col2:
        table_count = results.get("tables", {}).get("count", 0)
        st.metric("🗃️ Tables Found", table_count, help="Number of tables identified")

    with col3:
        analysis_type = results.get("analysis_type", "full")
        st.metric(
            "🔍 Analysis Type", analysis_type.title(), help="Type of analysis performed"
        )

    with col4:
        timestamp = results.get("timestamp", "Unknown")
        st.metric(
            "⏰ Completed",
            timestamp.split()[1][:5] if " " in timestamp else "Now",
            help="Analysis completion time",
        )


def display_results_dashboard():
    """Display the main results dashboard."""

    if "migration_results" not in st.session_state:
        return

    results = st.session_state.migration_results

    st.markdown("---")
    st.markdown("## 📈 Migration Analysis Results")

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Overview",
            "📓 Notebooks",
            "🗃️ Tables",
            "�️ Jobs",
            "� Cost Analysis",
        ]
    )

    with tab1:
        display_overview_tab(results)

    with tab2:
        display_notebooks_tab(results)

    with tab3:
        display_tables_tab(results)

    with tab4:
        display_jobs_tab(results)

    with tab5:
        display_cost_analysis_tab(results)


def display_overview_tab(results):
    """Display overview tab content."""

    st.markdown("### 🎯 Migration Overview")

    # Summary metrics
    display_analysis_summary()

    # Risk summary and migration readiness
    notebook_analyses = results.get("notebook_analyses", [])
    if notebook_analyses:
        risk_counts = {"green": 0, "yellow": 0, "red": 0}
        auto_migrate = 0
        needs_review = 0
        for item in notebook_analyses:
            analysis = item.get("analysis", {})
            risk = analysis.get("risk", "green")
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

            # Decide auto-migrate vs needs review heuristically
            if analysis.get("complexity") == "simple" and not analysis.get("issues"):
                auto_migrate += 1
            else:
                needs_review += 1

        st.markdown("### ⚖️ Risk & Migration Readiness")
        col1, col2, col3 = st.columns(3)
        col1.metric("Green (low risk)", risk_counts.get("green", 0))
        col2.metric("Yellow (medium risk)", risk_counts.get("yellow", 0))
        col3.metric("Red (high risk)", risk_counts.get("red", 0))

        st.markdown("### ✅ Migration Summary")
        st.write(f"Auto-migrate candidates: **{auto_migrate}**")
        st.write(f"Requires manual review: **{needs_review}**")

        # Build detailed lists for auto-migrate vs needs-review
        import json

        import pandas as pd

        rows = []
        for item in notebook_analyses:
            nb = item.get("notebook") or {}
            analysis = item.get("analysis") or {}
            path = nb.get("path") or nb.get("object_path") or nb.get("name") or ""
            issues = analysis.get("issues") or []
            if isinstance(issues, str):
                # single string
                issues_list = [issues] if issues else []
            elif isinstance(issues, list):
                issues_list = issues
            else:
                issues_list = []

            size_kb = round((nb.get("size", 0) or 0) / 1024, 2)
            confidence_pct, confidence_label = compute_confidence_score(
                analysis.get("complexity"), len(issues_list), size_kb
            )

            rows.append(
                {
                    "path": path,
                    "name": (path.split("/")[-1] if path else ""),
                    "owner": nb.get("owner") or nb.get("username") or "",
                    "complexity": analysis.get("complexity"),
                    "risk": analysis.get("risk"),
                    "issues": ", ".join(issues_list) if issues_list else "",
                    "suggestions": (
                        ", ".join(
                            analysis.get("suggestions", [])
                            if isinstance(analysis.get("suggestions"), list)
                            else [analysis.get("suggestions")]
                        )
                        if analysis.get("suggestions")
                        else ""
                    ),
                    "size_kb": size_kb,
                    "confidence_pct": confidence_pct,
                    "confidence_label": confidence_label,
                }
            )

        df_all = pd.DataFrame(rows)

        df_auto = df_all[(df_all["complexity"] == "simple") & (df_all["issues"] == "")]
        df_review = df_all.drop(df_auto.index)

        # Show expandable sections with lists and download options
        with st.expander(f"Auto-migrate ({len(df_auto)})", expanded=False):
            if not df_auto.empty:
                st.dataframe(df_auto, use_container_width=True)
                csv_data = df_auto.to_csv(index=False)
                json_data = df_auto.to_json(orient="records", indent=2)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download CSV",
                        data=csv_data,
                        file_name="auto_migrate_notebooks.csv",
                    )
                with col2:
                    st.download_button(
                        "Download JSON",
                        data=json_data,
                        file_name="auto_migrate_notebooks.json",
                    )
            else:
                st.info("No clear auto-migrate candidates found.")

        with st.expander(f"Needs manual review ({len(df_review)})", expanded=True):
            if not df_review.empty:
                st.dataframe(df_review, use_container_width=True)
                csv_data = df_review.to_csv(index=False)
                json_data = df_review.to_json(orient="records", indent=2)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download CSV",
                        data=csv_data,
                        file_name="needs_review_notebooks.csv",
                    )
                with col2:
                    st.download_button(
                        "Download JSON",
                        data=json_data,
                        file_name="needs_review_notebooks.json",
                    )
            else:
                st.info("All notebooks appear auto-migratable by current heuristics.")

        # Inventory download (full)
        inventory = {
            "notebooks": [n.get("notebook") for n in notebook_analyses],
            "notebook_analyses": [n.get("analysis") for n in notebook_analyses],
            "tables": results.get("tables", {}).get("tables", []),
            "jobs": results.get("jobs", {}).get("jobs", []),
        }

        if st.download_button(
            "Download JSON inventory",
            data=json.dumps(inventory, indent=2),
            file_name="inventory.json",
        ):
            st.success("Inventory downloaded")

    # Recommendations
    st.markdown("### 💡 Key Recommendations")

    recommendations = [
        "🔄 Start with low-complexity notebooks for initial migration",
        "📊 Consider EMR Serverless for variable workloads",
        "💾 Use S3 Intelligent Tiering for cost optimization",
        "🔍 Review high-complexity notebooks manually",
        "📈 Monitor costs during migration process",
    ]

    for rec in recommendations:
        st.info(rec)


def display_notebooks_tab(results):
    """Display notebooks analysis tab."""

    st.markdown("### 📓 Notebook Analysis")

    # Ensure we have the current app configuration available here (sidebar values)
    app_config = get_configuration()

    notebooks_data = results.get("notebooks", {})
    notebooks = notebooks_data.get("notebooks", [])

    if not notebooks:
        st.warning("No notebooks found or analyzed.")
        return

    # Display notebooks table
    import pandas as pd

    # Build notebooks DataFrame with extra fields: Owner and Risk & Migration
    def _username_from_path(p: str) -> str:
        if not p:
            return "unknown"
        try:
            parts = p.strip("/").split("/")
            if parts[0] == "Users" and len(parts) >= 2:
                return parts[1]
            if parts[0] == "Shared":
                return "Shared"
            return parts[0]
        except Exception:
            return "unknown"

    notebook_analyses = results.get("notebook_analyses", []) or []
    notebook_analyses_map = {}
    for item in notebook_analyses:
        if not isinstance(item, dict):
            continue
        nb_meta = item.get("notebook") or {}
        analysis = item.get("analysis") or {}
        path = nb_meta.get("path") or nb_meta.get("object_path") or nb_meta.get("name")
        if path:
            notebook_analyses_map[path] = analysis

    df = pd.DataFrame(
        [
            {
                "Path": nb.get("path", "Unknown"),
                "Name": (
                    (nb.get("path", "Unknown").split("/")[-1])
                    if nb.get("path")
                    else "Unknown"
                ),
                "Language": nb.get("language", "Unknown"),
                # Prefer explicit owner metadata returned by the connector
                "Owner": nb.get("owner") or _username_from_path(nb.get("path", "")),
                "Risk & Migration": notebook_analyses_map.get(
                    nb.get("path", ""), {}
                ).get("risk", "unknown"),
                "Last Modified": (
                    (
                        (
                            pd.to_datetime(
                                nb.get("modified_at")
                                or nb.get("updated_at")
                                or nb.get("modified")
                                or nb.get("last_modified"),
                                errors="coerce",
                            )
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        if pd.notna(
                            pd.to_datetime(
                                nb.get("modified_at")
                                or nb.get("updated_at")
                                or nb.get("modified")
                                or nb.get("last_modified"),
                                errors="coerce",
                            )
                        )
                        else ""
                    ),
                ),
                "Size (KB)": round(nb.get("size", 0) / 1024, 2),
                "Lines": nb.get("content", "").count("\n") if nb.get("content") else 0,
                # compute confidence from analysis if available
                **(
                    (
                        lambda a, s: {
                            # Bedrock status summary (used/model/error/not enabled)
                            "Bedrock": (
                                (
                                    lambda b: (
                                        "Used: {}".format(b.get("model_id"))
                                        if b and b.get("ok")
                                        else (
                                            "Error: {}".format(b.get("error"))
                                            if b and b.get("error")
                                            else "Not enabled"
                                        )
                                    )
                                )(
                                    notebook_analyses_map.get(
                                        nb.get("path", ""), {}
                                    ).get("bedrock_enrichment")
                                )
                            ),
                            "confidence_pct": compute_confidence_score(
                                a.get("complexity"),
                                len(a.get("issues") or []),
                                round((nb.get("size", 0) or 0) / 1024, 2),
                            )[0],
                            "confidence_label": compute_confidence_score(
                                a.get("complexity"),
                                len(a.get("issues") or []),
                                round((nb.get("size", 0) or 0) / 1024, 2),
                            )[1],
                        }
                    )
                    if (lambda: True)()
                    else {}
                )(notebook_analyses_map.get(nb.get("path", ""), {}), nb.get("size", 0)),
            }
            for nb in notebooks
        ]
    )

    # Search and filters (language and risk)
    search_q = st.text_input("Search notebooks (path, name or owner)")
    languages = sorted(df["Language"].dropna().unique().tolist())
    selected_language = st.selectbox(
        "Filter by Language", options=["All"] + languages, index=0
    )

    risks = sorted(df["Risk & Migration"].dropna().unique().tolist())
    selected_risk = st.selectbox("Filter by Risk", options=["All"] + risks, index=0)

    # Apply filters
    df_filtered = df.copy()
    if selected_language != "All":
        df_filtered = df_filtered[df_filtered["Language"] == selected_language]
    if selected_risk != "All":
        df_filtered = df_filtered[df_filtered["Risk & Migration"] == selected_risk]
    if search_q:
        q = search_q.lower()
        df_filtered = df_filtered[
            df_filtered["Path"].str.lower().str.contains(q)
            | df_filtered["Name"].str.lower().str.contains(q)
            | df_filtered["Owner"].str.lower().str.contains(q)
        ]

    st.dataframe(df_filtered, width="stretch")

    # Show sample notebook content
    if not df_filtered.empty:
        st.markdown("### 📝 Sample Notebook Content")
        # Use the filtered list for selection
        filtered_notebooks = [
            notebooks[i]
            for i, nb in enumerate(notebooks)
            if (
                (selected_language == "All" or nb.get("language") == selected_language)
                and (
                    selected_risk == "All"
                    or nb.get("analysis", {}).get("risk") == selected_risk
                )
                and (
                    not search_q
                    or search_q.lower() in (nb.get("path", "") or "").lower()
                    or search_q.lower()
                    in (nb.get("owner", nb.get("username", "")) or "").lower()
                )
            )
        ]

        options = [
            nb.get("path", f"Notebook {i}") for i, nb in enumerate(filtered_notebooks)
        ]
        selected_notebook = st.selectbox("Select notebook to preview:", options=options)

        if selected_notebook:
            notebook_idx = next(
                (
                    i
                    for i, nb in enumerate(filtered_notebooks)
                    if nb.get("path") == selected_notebook
                ),
                0,
            )
            content = filtered_notebooks[notebook_idx].get(
                "content", "No content available"
            )
            st.code(
                content[:1000] + "..." if len(content) > 1000 else content,
                language="python",
            )

            # Notebook conversion actions (template-based)
            st.markdown("---")
            from converters.notebook_converter import NotebookConverter

            conv_bucket = st.text_input(
                "Destination S3 bucket for converted notebooks",
                value=app_config.get("s3_bucket") or "",
            )

            notebook_converter = NotebookConverter(config)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Convert Selected Notebook to S3"):
                    with st.spinner("Converting and uploading..."):
                        nb = filtered_notebooks[notebook_idx]
                        res = notebook_converter.convert_notebook(
                            nb.get("content", ""),
                            nb.get("path", "unnamed.py"),
                            language=nb.get("language", "PYTHON"),
                            bucket=conv_bucket or None,
                        )
                        if res.get("uploaded"):
                            st.success(
                                f"Uploaded to s3://{conv_bucket}/{res.get('converted_key')}"
                            )
                        else:
                            st.error("Upload failed; check logs and permissions")
                        # show a small diff preview
                        diffs = res.get("diffs", [])[:200]
                        st.text("Diff preview (first 200 ops):")
                        st.code(
                            "\n".join([f"{op} {line.rstrip()}" for op, line in diffs]),
                            language="text",
                        )

            with col2:
                if st.button("Convert All Visible Notebooks to S3"):
                    notebooks_to_convert = [
                        {
                            "path": nb.get("path"),
                            "content": nb.get("content"),
                            "language": nb.get("language", "PYTHON"),
                        }
                        for nb in filtered_notebooks
                    ]
                    with st.spinner(
                        f"Converting {len(notebooks_to_convert)} notebooks..."
                    ):
                        batch_res = notebook_converter.convert_notebooks_batch(
                            notebooks_to_convert, bucket=conv_bucket or None
                        )
                        success_count = sum(1 for r in batch_res if r.get("uploaded"))
                        st.success(
                            f"Uploaded {success_count} / {len(batch_res)} notebooks to s3://{conv_bucket}"
                        )
                        # Provide a downloadable JSON report
                        report_json = json.dumps(batch_res, indent=2)
                        st.download_button(
                            "Download conversion report (JSON)",
                            data=report_json,
                            file_name="conversion_report.json",
                        )


def display_tables_tab(results):
    """Display tables analysis tab."""

    st.markdown("### 🗃️ Table Analysis")

    tables_data = results.get("tables", {})
    tables = tables_data.get("tables", [])

    if not tables:
        st.warning("No tables found.")
        return

    # Display tables information with quick catalog/schema filters
    import pandas as pd

    # Build initial DataFrame
    df_all = pd.DataFrame(
        [
            {
                "Name": table.get("name", "Unknown"),
                "Catalog": table.get("catalog", "Unknown"),
                "Schema": table.get("schema", "Unknown"),
                "Format": table.get("data_source_format", "Unknown"),
                "Location": table.get("storage_location")
                or table.get("location")
                or "",
                "Size (MB)": round(table.get("size_bytes", 0) / (1024 * 1024), 2),
                "SizeBytes": table.get("size_bytes", 0),
                "Columns": len(table.get("columns", [])),
            }
            for table in tables
        ]
    )

    # Quick filters: catalog and schema
    catalogs = sorted(df_all["Catalog"].dropna().unique().tolist())
    selected_catalog = st.selectbox(
        "Filter by Catalog", options=["All"] + catalogs, index=0
    )

    if selected_catalog != "All":
        df_catalog = df_all[df_all["Catalog"] == selected_catalog]
    else:
        df_catalog = df_all

    schemas = sorted(df_catalog["Schema"].dropna().unique().tolist())
    selected_schema = st.selectbox(
        "Filter by Schema", options=["All"] + schemas, index=0
    )

    if selected_schema != "All":
        df_filtered = df_catalog[df_catalog["Schema"] == selected_schema]
    else:
        df_filtered = df_catalog

    st.dataframe(df_filtered.drop(columns=["SizeBytes"]), width="stretch")

    # Show table details
    if tables:
        st.markdown("### 📋 Table Details")
        # Use the filtered list of tables for selection
        filtered_tables = [
            t
            for t in tables
            if (
                (selected_catalog == "All" or t.get("catalog") == selected_catalog)
                and (selected_schema == "All" or t.get("schema") == selected_schema)
            )
        ]

        # Use fully-qualified table names in the selectbox to avoid ambiguity: catalog.schema.name
        fq_names = [
            f"{t.get('catalog', 'unknown')}.{t.get('schema', 'default')}.{t.get('name', f'Table{i}') }"
            for i, t in enumerate(filtered_tables)
        ]

        selected_table = st.selectbox("Select table to view details:", options=fq_names)

        if selected_table:
            # Resolve selected fully-qualified name back to table index
            def _parse_fq_name(fq: str):
                parts = fq.split(".")
                if len(parts) >= 3:
                    catalog = parts[0]
                    schema = parts[1]
                    name = ".".join(parts[2:])
                elif len(parts) == 2:
                    catalog = parts[0]
                    schema = parts[1]
                    name = ""
                else:
                    catalog = ""
                    schema = ""
                    name = parts[0]
                return catalog, schema, name

            sel_catalog, sel_schema, sel_name = _parse_fq_name(selected_table)

            table_idx = next(
                (
                    i
                    for i, table in enumerate(tables)
                    if table.get("name") == sel_name
                    and table.get("schema") == sel_schema
                    and table.get("catalog") == sel_catalog
                ),
                0,
            )

            table_info = tables[table_idx]

            col1, col2 = st.columns(2)

            with col1:
                st.json(
                    {
                        "name": table_info.get("name"),
                        "catalog": table_info.get("catalog"),
                        "schema": table_info.get("schema"),
                        "table_type": table_info.get("table_type"),
                        "format": table_info.get("data_source_format"),
                        "location": table_info.get("storage_location")
                        or table_info.get("location")
                        or "",
                    }
                )

            with col2:
                columns = table_info.get("columns", [])
                # If types are missing or columns empty, attempt to fetch detailed column info
                need_fetch = False
                if not columns:
                    need_fetch = True
                else:
                    # detect if any column is missing a 'type' key or has None
                    for c in columns:
                        if not c.get("type"):
                            need_fetch = True
                            break

                # Attempt to use a live connector to fetch column types if needed
                fetched_columns = []
                if need_fetch:
                    try:
                        # Initialize a temporary connector using current sidebar config
                        temp_tool = initialize_migration_tool(app_config)
                        fetched_columns = (
                            temp_tool.databricks_connector.get_table_columns(
                                table_info.get("catalog") or "",
                                table_info.get("schema") or "",
                                table_info.get("name") or "",
                            )
                        )
                    except Exception:
                        fetched_columns = []

                display_columns = fetched_columns if fetched_columns else columns

                if display_columns:
                    st.markdown("**Columns:**")
                    for col in display_columns[:50]:  # Show up to 50 columns safely
                        st.text(
                            f"• {col.get('name', 'Unknown')} ({col.get('type', 'Unknown')})"
                        )

                    if len(display_columns) > 50:
                        st.text(f"... and {len(display_columns) - 50} more columns")
                else:
                    st.info("No column metadata available for this table.")


def display_jobs_tab(results):
    """Display simple Databricks jobs (name, type, run_as user, trigger)."""

    st.markdown("### 🛠️ Jobs Analysis")

    jobs_data = results.get("jobs", {})
    jobs = jobs_data.get("jobs") if isinstance(jobs_data, dict) else jobs_data

    if not jobs:
        st.info("No jobs discovered. Run the analysis to populate jobs.")
        return

    import json

    import pandas as pd

    def _job_type_from_settings(settings: dict) -> str:
        # Inspect settings to determine simple job type (notebooks, spark-submit, jar, pythonWheel, dbt, etc.)
        if not settings:
            return "unknown"
        if settings.get("notebook_task"):
            return "notebook"
        if settings.get("spark_jar_task"):
            return "jar"
        if settings.get("spark_python_task"):
            return "python"
        if settings.get("spark_submit_task"):
            return "spark-submit"
        if settings.get("dbt_task"):
            return "dbt"
        return "other"

    def _trigger_from_job(job: dict) -> str:
        # Simple heuristics: schedule => scheduled, file_arrival => file-arrival, paused => paused, else ad-hoc
        settings = job.get("settings", {})
        schedule = settings.get("schedule")
        if isinstance(schedule, dict):
            # scheduled if a cron or periodic pattern exists
            if schedule.get("quartz_cron_expression") or schedule.get("cron"):
                return "scheduled"
        # Some jobs use "file_arrival_notification" in settings or tasks
        try:
            if "file_arrival_notification" in json.dumps(settings).lower():
                return "file-arrival"
        except Exception:
            pass
        # paused if job is disabled or has a paused state
        if job.get("settings", {}).get("email_notifications", {}) is None and job.get(
            "is_paused"
        ):
            return "paused"
        return "ad-hoc"

    rows = []
    for j in jobs:
        settings = j.get("settings") or {}
        rows.append(
            {
                "Job ID": j.get("job_id") or j.get("id"),
                "Name": (
                    settings.get("name") or j.get("name") or f"job_{j.get('job_id')}"
                ),
                "Type": _job_type_from_settings(settings),
                "Run As": (
                    settings.get("run_as_user")
                    or settings.get("run_as")
                    or j.get("creator_user_name")
                    or j.get("created_by")
                ),
                "Trigger": _trigger_from_job(j),
            }
        )

    df = pd.DataFrame(rows)

    st.dataframe(df, width="stretch")

    # Provide a simple JSON view for the selected job
    selected = st.selectbox("Select job to inspect:", options=df["Name"].tolist())
    if selected:
        sel_idx = df.index[df["Name"] == selected].tolist()
        if sel_idx:
            job_obj = jobs[sel_idx[0]]
            st.markdown("### 📄 Job Details")
            st.json(job_obj)


def display_cost_analysis_tab(results):
    """Display cost analysis tab."""

    st.markdown("### 💰 Cost Analysis")

    import pandas as pd

    tables = results.get("tables", {}).get("tables", [])
    clusters = results.get("clusters", {}).get("clusters", [])

    if not tables and not clusters:
        st.info(
            "No table or cluster data available to compute TCO. Run analysis first."
        )
        return

    # Show storage breakdown by catalog (if tables exist)
    if tables:
        df = pd.DataFrame(
            [
                {
                    "catalog": t.get("catalog", "unknown"),
                    "size_bytes": t.get("size_bytes", 0),
                }
                for t in tables
            ]
        )

        if not df.empty:
            agg = df.groupby("catalog")["size_bytes"].sum().reset_index()
            agg["size_gb"] = agg["size_bytes"] / (1024**3)

            # Use configured storage price if available
            storage_cost_per_gb_month = (
                config_manager.config.get("costs", {})
                .get("aws", {})
                .get("s3_standard_gb_month", 0.023)
            )
            agg["monthly_storage_cost_usd"] = agg["size_gb"] * storage_cost_per_gb_month

            st.markdown("#### Estimated Monthly Storage Cost by Catalog")
            st.dataframe(
                agg[["catalog", "size_gb", "monthly_storage_cost_usd"]].rename(
                    columns={
                        "catalog": "Catalog",
                        "size_gb": "Size (GB)",
                        "monthly_storage_cost_usd": "Monthly Storage Cost (USD)",
                    }
                ),
                use_container_width=True,
            )

            try:
                chart_df = agg.set_index("catalog")["monthly_storage_cost_usd"]
                st.bar_chart(chart_df)
            except Exception:
                pass

            total_monthly = agg["monthly_storage_cost_usd"].sum()
            total_annual = total_monthly * 12

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Estimated Monthly Storage Cost", f"${total_monthly:,.2f}")
            with col2:
                st.metric("Estimated Annual Storage Cost", f"${total_annual:,.2f}")

    # Use the TCO helpers for a fuller cost comparison
    from tco.tco import (
        add_buffer,
        bedrock_recommendations_placeholder,
        estimate_databricks_costs_from_clusters,
        estimate_storage_costs_from_tables,
        generate_simple_roi,
        map_to_aws_emr_estimate,
    )

    # Estimate DB compute (30-day lookback)
    db_compute = estimate_databricks_costs_from_clusters(
        clusters, lookback_days=30, config=config_manager
    )

    storage = estimate_storage_costs_from_tables(tables, config=config_manager)

    # Map to AWS EMR estimate
    aws_emr = map_to_aws_emr_estimate(db_compute, config=config_manager)

    # Add 15% buffer
    db_compute_buffered = add_buffer(db_compute.get("compute_cost_usd", 0))
    db_storage_buffered = add_buffer(storage.get("monthly_storage_cost_usd", 0))

    aws_compute_buffered = add_buffer(aws_emr.get("compute_cost_usd", 0))
    aws_storage_buffered = add_buffer(storage.get("monthly_storage_cost_usd", 0))

    # Side-by-side comparison
    comp = {
        "Metric": ["Compute (monthly)", "Storage (monthly)", "Total Monthly"],
        "Databricks (USD)": [
            f"${db_compute_buffered:,.2f}",
            f"${db_storage_buffered:,.2f}",
            f"${(db_compute_buffered + db_storage_buffered):,.2f}",
        ],
        "AWS (USD)": [
            f"${aws_compute_buffered:,.2f}",
            f"${aws_storage_buffered:,.2f}",
            f"${(aws_compute_buffered + aws_storage_buffered):,.2f}",
        ],
    }

    comp_df = pd.DataFrame(comp)
    st.markdown("#### Side-by-side cost comparison (with 15% buffer)")
    st.dataframe(comp_df, width="stretch")

    # Projections & ROI
    db_monthly = db_compute_buffered + db_storage_buffered
    aws_monthly = aws_compute_buffered + aws_storage_buffered
    roi = generate_simple_roi(db_monthly, aws_monthly)

    st.markdown("#### Projections & ROI")
    st.markdown(f"- Databricks monthly (buffered): **${db_monthly:,.2f}**")
    st.markdown(f"- AWS monthly (buffered): **${aws_monthly:,.2f}**")
    st.markdown(f"- Estimated monthly savings: **${roi['monthly_savings_usd']:,.2f}**")
    st.markdown(f"- Estimated annual savings: **${roi['annual_savings_usd']:,.2f}**")
    if roi.get("months_to_recover"):
        st.markdown(
            f"- Estimated months to recover migration cost: **{roi['months_to_recover']:.1f} months**"
        )

    # AI-based cost optimization suggestions (Bedrock) — best-effort and guarded
    st.markdown("#### AI Recommendations: Cost Optimization")
    try:
        tips = bedrock_recommendations_placeholder(db_compute)
        for t in tips:
            st.info(t)
    except Exception as e:
        logger.error(f"Bedrock recommendation error: {e}", exc_info=True)


def load_results(migration_tool):
    """Load previous migration results."""

    try:
        results_dir = Path("data/results")

        if not results_dir.exists():
            st.warning("⚠️ No previous results found.")
            return

        # List available result files
        result_files = list(results_dir.glob("migration_results_*.json"))

        if not result_files:
            st.warning("⚠️ No previous results found.")
            return

        # Let user select file to load
        selected_file = st.selectbox(
            "Select results file to load:",
            options=[f.name for f in sorted(result_files, reverse=True)],
            key="load_results_select",
        )

        if selected_file and st.button("📂 Load Selected Results", width="stretch"):
            filepath = results_dir / selected_file

            with open(filepath, "r") as f:
                loaded_results = json.load(f)

            st.session_state.migration_results = loaded_results
            st.success(f"✅ Results loaded from {selected_file}")
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error loading results: {e}")
        logger.error(f"Load results error: {e}", exc_info=True)


def show_sample_data():
    """Show sample data for demonstration."""

    st.markdown("### 📋 Sample Data")

    sample_notebook = """
# Databricks notebook source
import pandas as pd
from pyspark.sql import SparkSession
from delta.tables import DeltaTable

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM delta.`/databricks-datasets/nyctaxi/tables/nyctaxi_yellow` LIMIT 10

# COMMAND ----------

# Use dbutils to list files
files = dbutils.fs.ls("/databricks-datasets/")
display(files)

# COMMAND ----------

# Create Delta table
df = spark.read.format("delta").load("/path/to/delta/table")
display(df.limit(100))
    """

    st.code(sample_notebook, language="python")


def show_documentation():
    """Show documentation and help."""

    st.markdown(
        """
    ### 📖 Documentation

    #### 🚀 Getting Started
    1. Configure your Databricks and AWS credentials
    2. Test connections to ensure everything works
    3. Run analysis to discover notebooks and tables
    4. Review results and migration recommendations

    #### 🔧 Configuration
    - **Databricks**: Workspace URL and access token required
    - **AWS**: Region and S3 bucket for data storage
    - **Bedrock**: AI model for code analysis

    #### 📊 Analysis Types
    - **Full Analysis**: Complete migration assessment
    - **Quick Analysis**: Notebook discovery only

    #### 💡 Tips
    - Start with a small subset of notebooks
    - Review high-complexity items manually
    - Test converted code in EMR environment
    - Monitor costs during migration

    #### 🆘 Support
    - GitHub Issues: Report bugs and feature requests
    - Documentation: Detailed setup and usage guides
    - Community: Join discussions and share experiences
    """
    )


def show_getting_started_guide():
    """Show getting started guide when no results are available."""

    st.markdown("---")
    st.markdown("## 🚀 Getting Started Guide")

    st.markdown(
        """
    <div class="feature-card">
    <h3>🎯 Ready to migrate your Databricks workloads?</h3>
    <p>Follow these steps to get started with your migration analysis:</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### 1️⃣ Configure Settings
        - ✅ Set Databricks workspace URL
        - ✅ Add your access token
        - ✅ Configure AWS region and S3 bucket
        - ✅ Test all connections
        """
        )

        st.markdown(
            """
        ### 3️⃣ Review Results
        - 📊 Analyze complexity scores
        - 💰 Review cost projections
        - 🔄 Examine code conversions
        - 📋 Plan migration strategy
        """
        )

    with col2:
        st.markdown(
            """
        ### 2️⃣ Run Analysis
        - 🚀 Full Analysis: Complete assessment
        - 📊 Quick Analysis: Notebook discovery
        - ⏱️ Wait for AI-powered analysis
        - 💾 Save results for later
        """
        )

        st.markdown(
            """
        ### 4️⃣ Execute Migration
        - 🔄 Convert notebooks to EMR format
        - 📦 Migrate tables to S3
        - ✅ Validate data integrity
        - 🎉 Celebrate your success!
        """
        )

    # Sample workflow
    st.markdown("### 📋 Sample Workflow")

    workflow_steps = [
        "🔧 Complete sidebar configuration",
        "🔗 Test all connections",
        "🚀 Run Full Analysis",
        "📊 Review analysis results",
        "💾 Save results for reference",
        "🔄 Begin migration process",
    ]

    for i, step in enumerate(workflow_steps, 1):
        st.markdown(f"**Step {i}:** {step}")


if __name__ == "__main__":
    # Add pandas import for timestamp functionality
    import json

    import pandas as pd

    main()
