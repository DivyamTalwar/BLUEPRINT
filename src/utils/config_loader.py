import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dotenv import load_dotenv


class ConfigLoader:

    def __init__(self, config_path: Optional[str] = None):
        # Load environment variables
        load_dotenv()

        # Default config path
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation)

        Args:
            key: Configuration key (e.g., 'api.openai.model')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration"""
        return self.config

    def validate_env_vars(self) -> Dict[str, bool]:
        """
        Validate required environment variables

        Returns:
            Dict with validation results
        """
        required_vars = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "PINECONE_API_KEY",
        ]

        validation = {}
        for var in required_vars:
            value = os.getenv(var)
            validation[var] = bool(value and value != "your_" + var.lower())

        return validation

    def get_stage_config(self, stage: int) -> Dict[str, Any]:
        """
        Get configuration for specific stage

        Args:
            stage: Stage number (1, 2, or 3)

        Returns:
            Stage configuration
        """
        return self.config.get(f"stage{stage}", {})

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.config.get("api", {})

    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration"""
        return self.config.get("vector_db", {})

    def get_feature_tree_config(self) -> Dict[str, Any]:
        """Get feature tree configuration"""
        return self.config.get("feature_tree", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config.get("logging", {})

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.config.get("performance", {})

    def get_cost_config(self) -> Dict[str, Any]:
        """Get cost tracking configuration"""
        return self.config.get("cost", {})

    def reload(self):
        """Reload configuration from file"""
        self.config = self._load_config()


# Global config instance
_config_instance: Optional[ConfigLoader] = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Get global configuration instance

    Args:
        config_path: Optional path to config file

    Returns:
        ConfigLoader instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)

    return _config_instance
