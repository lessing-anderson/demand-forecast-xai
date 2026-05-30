"""
Configuration loader for experiments.
"""
import yaml
from pathlib import Path


def load_config(config_path):
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Dictionary with configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config, output_path):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Config saved to {output_path}")
