import json
import os
from typing import Any, Dict, Optional

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json file."""
    config_path = "config.json"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation (e.g., 'api.openrouter_api_key').
    
    Args:
        key_path: Dot-separated path to the config value
        default: Default value if key not found
    
    Returns:
        Configuration value or default
    """
    try:
        config = load_config()
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            value = value[key]
        
        return value
    except (KeyError, FileNotFoundError):
        return default

def get_openrouter_api_key() -> Optional[str]:
    """Get the OpenRouter API key from config."""
    return get_config_value('llm.openrouter_api_key')
