import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import pandas as pd
import torch
import yaml
import numpy as np

logger = logging.getLogger(__name__)

def _json_default(obj):
    """Helper for JSON serialization of numpy types."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, torch.Tensor):
        obj = obj.cpu()
        return obj.item() if obj.numel() == 1 else obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def _sanitize_for_json(obj):
    """Recursively converts Tensors and other non-JSON-serializable objects to standard Python types..."""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, (torch.Tensor, np.integer, np.floating)):
                if isinstance(k, torch.Tensor):
                    k = k.item()
                else:
                    k = k.item()

            new_dict[k] = _sanitize_for_json(v)
        return new_dict
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, set):
        return [_sanitize_for_json(v) for v in list(obj)]
    elif isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu()
        if obj.numel() == 1:
            return obj.item()
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def calculate_config_hash(config: Dict[str, Any]) -> str:
    """Computes a deterministic SHA256 hash of the configuration dictionary."""
    def _sort_dict(d):
        if isinstance(d, dict):
            return {k: _sort_dict(v) for k, v in sorted(d.items())}
        if isinstance(d, list):
            return [_sort_dict(v) for v in d]
        return d

    sorted_config = _sort_dict(config)
    # separators=(',', ':') removes whitespace to make it more compact and stable
    config_str = json.dumps(sorted_config, sort_keys=True, default=_json_default, separators=(',', ':'))
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()

class ExperimentManager:
    """Manages experiment artifacts, configurations, and registry."""
    
    def __init__(self, base_path: Union[str, Path] = "artifacts"):
        self.base_path = Path(base_path)
        self.experiments_dir = self.base_path / "experiments"
        self.registry_path = self.base_path / "registry.json"
        
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
    def get_experiment_path(self, config: Dict[str, Any]) -> Path:
        """Returns the path for a given config, creating the hash internally."""
        exp_id = calculate_config_hash(config)
        return self.experiments_dir / exp_id

    def list_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Returns the content of the registry."""
        if not self.registry_path.exists():
            return {}
        try:
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode registry at {self.registry_path}")
            return {}

    def _update_registry(self, exp_id: str, config: Dict[str, Any]):
        """Updates the registry.json with the new experiment."""
        registry = self.list_experiments()

        dataset_config = config.get("data_preparation", {}).get("dataset", {})
        unlearning_config = config.get("data_preparation", {}).get("unlearning_set", {})
        
        entry = {
            "id": exp_id,
            "dataset": dataset_config.get("name"),
            "dataset_seed": dataset_config.get("seed"),
            "unlearning_type": unlearning_config.get("type"),
            "created_at": pd.Timestamp.now().isoformat(),
            "path": str(self.experiments_dir / exp_id)
        }
        
        registry[exp_id] = entry
        
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2, default=_json_default)

    def _create_dataset_fingerprint(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a fingerprint of the dataset configuration."""
        data_prep = config.get("data_preparation", {})
        dataset = data_prep.get("dataset", {})
        unlearning = data_prep.get("unlearning_set", {})
        
        fingerprint = {
            "dataset_name": dataset.get("name"),
            "train_subset": dataset.get("train_subset"),
            "test_subset": dataset.get("test_subset"),
            "seed": dataset.get("seed"),
            "unlearning_type": unlearning.get("type"),
            "num_forget": unlearning.get("num_forget"),
            "data_config_hash": calculate_config_hash({"data_preparation": data_prep})
        }
        return fingerprint

    def initialize_experiment(self, config: Dict[str, Any]) -> Path:
        """Initializes the experiment directory structure."""
        exp_id = calculate_config_hash(config)
        exp_dir = self.experiments_dir / exp_id
        
        if exp_dir.exists():
            logger.info(f"Experiment {exp_id} already exists at {exp_dir}")
        else:
            exp_dir.mkdir(parents=True)
            logger.info(f"Initialized new experiment {exp_id} at {exp_dir}")
        
        (exp_dir / "stages").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)

        with open(exp_dir / "config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        fingerprint = self._create_dataset_fingerprint(config)
        with open(exp_dir / "dataset_fingerprint.json", 'w') as f:
            json.dump(fingerprint, f, indent=2, default=_json_default)

        self._update_registry(exp_id, config)
        
        return exp_dir

    def save_stage(
        self,
        exp_id: str,
        stage_name: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, Any]] = None,
        predictions: Optional[pd.DataFrame] = None
    ):
        """
        Saves artifacts for a specific stage (e.g., 'base', 'finetuned', 'unlearned').
        """
        stage_dir = self.experiments_dir / exp_id / "stages" / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        if model is not None:
            torch.save(model.state_dict(), stage_dir / "model.pt")

        if optimizer is not None:
            torch.save(optimizer.state_dict(), stage_dir / "optimizer.pt")

        if metrics is not None:
            metrics_safe = _sanitize_for_json(metrics)
            with open(stage_dir / "metrics.json", 'w') as f:
                json.dump(metrics_safe, f, indent=2, default=_json_default)

        if predictions is not None:
            predictions.to_csv(stage_dir / "predictions.csv", index=False)
            
        logger.info(f"Saved stage '{stage_name}' artifacts to {stage_dir}")

    def load_stage(
        self,
        exp_id: str,
        stage_name: str,
        model_class: Optional[Any] = None,
        optimizer_class: Optional[Any] = None,
        device: str = "cpu",
        **model_kwargs
    ) -> Dict[str, Any]:
        """Loads artifacts for a specific stage."""
        stage_dir = self.experiments_dir / exp_id / "stages" / stage_name
        if not stage_dir.exists():
            raise FileNotFoundError(f"Stage {stage_name} not found for experiment {exp_id}")
            
        results = {}

        model_path = stage_dir / "model.pt"
        if model_path.exists():
            if model_class is not None:
                model = model_class(**model_kwargs)
                model.load_state_dict(torch.load(model_path, map_location=device))
                results['model'] = model
            else:
                # No class provided, so return the raw state dict instead of an instantiated model
                results['model'] = torch.load(model_path, map_location=device)

        opt_path = stage_dir / "optimizer.pt"
        if opt_path.exists():
            results['optimizer_state_dict'] = torch.load(opt_path, map_location=device)

        metrics_path = stage_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                results['metrics'] = json.load(f)

        preds_path = stage_dir / "predictions.csv"
        if preds_path.exists():
            results['predictions'] = pd.read_csv(preds_path)
            
        return results

    def load_config(self, exp_id: str) -> Dict[str, Any]:
        """Loads the configuration for a given experiment ID."""
        config_path = self.experiments_dir / exp_id / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found for experiment {exp_id}")
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_by_hash(self, exp_id: str, stage_name: str) -> Dict[str, Any]:
        """Convenience method to load a specific stage by hash directly."""
        return self.load_stage(exp_id, stage_name)
