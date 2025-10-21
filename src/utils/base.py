"""
Base classes and utilities for the migration tool.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from pydantic import BaseModel, Field

    _PYDANTIC_AVAILABLE = True
except Exception:
    # Fallback lightweight implementations to allow importing the codebase
    _PYDANTIC_AVAILABLE = False

    class BaseModel:
        """Minimal BaseModel fallback when pydantic is not installed.

        This provides only a simple initializer and attribute access so
        other modules can import and run in environments without pydantic
        (useful for local editing/testing). It is NOT a replacement for
        pydantic features in production.
        """

        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)

    def Field(default=None, **kwargs):
        return default


class BaseConfig(BaseModel):
    """Base configuration class."""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


if _PYDANTIC_AVAILABLE:

    class MigrationResult(BaseModel):
        """Base result class for migration operations."""

        status: str = Field(..., description="Operation status")
        message: str = Field("", description="Status message")
        data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
        errors: List[str] = Field(default_factory=list, description="Error messages")
        warnings: List[str] = Field(
            default_factory=list, description="Warning messages"
        )

        @property
        def is_success(self) -> bool:
            """Check if operation was successful."""
            return self.status.lower() in ["success", "completed", "ok"]

        @property
        def has_errors(self) -> bool:
            """Check if operation has errors."""
            return len(self.errors) > 0

        @property
        def has_warnings(self) -> bool:
            """Check if operation has warnings."""
            return len(self.warnings) > 0

else:

    class MigrationResult:
        """Minimal MigrationResult fallback when pydantic is unavailable."""

        def __init__(
            self,
            status: str,
            message: str = "",
            data: Optional[Dict[str, Any]] = None,
            errors: Optional[List[str]] = None,
            warnings: Optional[List[str]] = None,
        ):
            self.status = status
            self.message = message
            self.data = data or {}
            self.errors = errors or []
            self.warnings = warnings or []

        @property
        def is_success(self) -> bool:
            return self.status.lower() in ["success", "completed", "ok"]

        @property
        def has_errors(self) -> bool:
            return len(self.errors) > 0

        @property
        def has_warnings(self) -> bool:
            return len(self.warnings) > 0


class BaseComponent(ABC):
    """Base class for all migration tool components."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize base component.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._setup()

    def _setup(self) -> None:
        """Setup component-specific configuration."""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate component configuration."""
        pass

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support.

        Args:
            key: Configuration key (supports dot notation like 'aws.region')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default


class ConfigManager:
    """Configuration manager for loading and managing configuration."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = {}
        self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, "r") as f:
                    self.config = yaml.safe_load(f)

                # Substitute environment variables
                self.config = self._substitute_env_vars(self.config)
            else:
                logging.warning(f"Configuration file not found: {self.config_path}")
                self.config = {}

        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            self.config = {}

        return self.config

    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute environment variables in configuration."""
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Handle ${VAR:default} syntax
            if obj.startswith("${") and obj.endswith("}"):
                var_expr = obj[2:-1]
                if ":" in var_expr:
                    var_name, default_value = var_expr.split(":", 1)
                    # Convert string boolean/numeric defaults
                    if default_value.lower() == "true":
                        default_value = True
                    elif default_value.lower() == "false":
                        default_value = False
                    elif default_value.isdigit():
                        default_value = int(default_value)
                    elif default_value.replace(".", "").isdigit():
                        default_value = float(default_value)
                else:
                    var_name = var_expr
                    default_value = None

                return os.getenv(var_name, default_value)
            return obj
        else:
            return obj

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self.config.update(updates)

    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file."""
        save_path = Path(path) if path else self.config_path

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")


class LoggerSetup:
    """Logger setup utility."""

    @staticmethod
    def setup_logging(config: Dict[str, Any]) -> logging.Logger:
        """
        Setup logging configuration.

        Args:
            config: Logging configuration

        Returns:
            Configured logger
        """
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Get logging configuration
        log_config = config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        log_format = log_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(logs_dir / "migration_tool.log"),
            ],
        )

        # Configure specific loggers
        loggers_config = log_config.get("loggers", {})
        for logger_name, logger_config in loggers_config.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(
                getattr(logging, logger_config.get("level", "INFO").upper())
            )

        return logging.getLogger("migration_tool")


# Global configuration instance
config_manager = ConfigManager()
