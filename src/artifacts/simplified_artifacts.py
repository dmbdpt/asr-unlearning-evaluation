import os
import json
import logging
import shutil
import tempfile
import time
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable

from omegaconf import DictConfig, OmegaConf
from pandas import DataFrame
import yaml
import mlflow
from mlflow import pytorch as mlflow_pytorch

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Experiment tracker using MLflow."""

    def __init__(self, cfg: DictConfig):
        self.experiment_name = cfg.experiment_name
        self.cache_dir = cfg.cache_dir
        self.tracking_uri = cfg.tracking_uri

        self._run = None
        self._run_id = None

        if not dist.is_initialized() or dist.get_rank() == 0:
            mlflow.set_tracking_uri(self.tracking_uri)
            self._mlflow_experiment = mlflow.set_experiment(experiment_name=self.experiment_name)

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> str:

        if mlflow.active_run() is not None and not nested:
            logger.warning("A run is already active. Ending the current run before starting a new one.")
            mlflow.end_run(status="KILLED")

        self._run = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
        self._run_id = self._run.info.run_id
        return self._run_id

    def end_run(self, status: str = "FINISHED"):
        mlflow.end_run(status=status)

    def log_params(self, params: Dict[str, Any], prefix: Optional[str] = None):
        if prefix:
            params = {f"{prefix}.{k}": v for k, v in params.items()}
        mlflow.log_params(params)

    def log_cfg(self, cfg: DictConfig, prefix: Optional[str] = None):
        flat_cfg = OmegaConf.to_container(cfg, resolve=True)

        def flatten(d, parent_key="", sep="."):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        params = flatten(flat_cfg)
        self.log_params(params, prefix=prefix)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None
    ):
        if prefix:
            metrics = {f"{prefix}.{k}": v for k, v in metrics.items()}
            sanitized_metrics = {}
            for key, value in metrics.items():
                sanitized_key = key.replace('@', '_at_').replace('=', '_eq_').replace('%', '_pct_')
                sanitized_metrics[sanitized_key] = value

            mlflow.log_metrics(sanitized_metrics, step=step)

    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None
    ):
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):

        artifact_dir = os.path.dirname(artifact_file) or None
        filename = os.path.basename(artifact_file)
        suffix = os.path.splitext(filename)[1] or '.json'

        tmp_path = None

        # OmegaConf's DictConfig isn't JSON-serializable directly
        if isinstance(dictionary, DictConfig):
            dictionary = OmegaConf.to_container(dictionary, resolve=True)
        elif isinstance(dictionary, dict):
            dictionary = self._convert_dictconfig_to_dict(dictionary)

        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', suffix=suffix, delete=False,
                    prefix=os.path.splitext(filename)[0] + '_') as f:
                json.dump(dictionary, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
                tmp_path = f.name

                mlflow.log_artifact(tmp_path, artifact_path=artifact_dir)
        except Exception as e:
            logger.error("Failed to log dict artifact: %s", e)
            
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def log_table(self, data: dict[str, Any] | DataFrame, artifact_file: str):
        mlflow.log_table(
            data=data,
            artifact_file=artifact_file
        )

    def _convert_dictconfig_to_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively convert any DictConfig in a dict to regular dict."""
        result = {}
        for k, v in d.items():
            if isinstance(v, DictConfig):
                result[k] = OmegaConf.to_container(v, resolve=True)
            elif isinstance(v, dict):
                result[k] = self._convert_dictconfig_to_dict(v)
            elif isinstance(v, list):
                result[k] = [OmegaConf.to_container(item, resolve=True) if isinstance(item, DictConfig) else
                             item for item in v]
            else:
                result[k] = v
        return result

    def log_model(self, model, **kwargs):
        mlflow_pytorch.log_model(model, **kwargs)

    def set_tags(self, tags: Dict[str, str]):
        mlflow.set_tags(tags)

    def get_run_id(self) -> Optional[str]:
        return self._run_id

class RankAwareTracker:
    def __init__(self, tracker, enabled: bool):
        self._tracker = tracker
        self._enabled = enabled

    def __getattr__(self, name):
        attr = getattr(self._tracker, name)

        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            if self._enabled:
                return attr(*args, **kwargs)
            return None

        return wrapper


class SimpleCache:
    """Simple disk-based caching for ML artifacts."""

    def __init__(
        self,
        cache_dir: str = ".cache",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.cache_dir / "cache_index.json"
        self.index: Dict[str, Dict[str, Any]] = {}
        self._load_index()

        logger.info("Cache initialized at %s", self.cache_dir)

    def _load_index(self):
        if self.index_path.exists():
            try:
                with open(self.index_path) as f:
                    self.index = json.load(f)
            except json.JSONDecodeError:
                self.index = {}

    def _save_index(self):
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    def _compute_key(
        self,
        artifact_type: str,
        config: DictConfig,
        suffix: Optional[str] = None
    ) -> str:
        config_dict = OmegaConf.to_container(config, resolve=True)
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        key_parts = [artifact_type, config_hash]
        if suffix:
            key_parts.append(suffix)
        return "_".join(key_parts)

    def get(
        self,
        artifact_type: str,
        config: DictConfig,
        suffix: Optional[str] = None,
        loader: Optional[Callable[[], Any]] = None
    ) -> tuple[Optional[Any], bool]:
        """Get from cache or compute."""
        key = self._compute_key(artifact_type, config, suffix)
        cache_path = self.cache_dir / artifact_type / f"{key}.pkl"

        if cache_path.exists() and key in self.index:
            try:
                with open(cache_path, 'rb') as f:
                    artifact = pickle.load(f)
                logger.info("Cache hit: %s", key)
                return artifact, True
            except Exception as e:
                logger.warning("Failed to load %s: %s", key, e)

        if loader is None:
            logger.info("Cache miss: %s", key)
            return None, False

        logger.info("Computing: %s", key)
        artifact = loader()

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(artifact, f)

                self.index[key] = {
                    'artifact_type': artifact_type,
                    'config': OmegaConf.to_container(config, resolve=True),
                    'created_at': datetime.now().isoformat(),
                    'path': str(cache_path)
                }
                self._save_index()
            except Exception as e:
                logger.warning("Failed to cache %s: %s", key, e)

        if dist.is_initialized():
            dist.barrier()

        return artifact, False

    def invalidate(self, artifact_type: str, config: DictConfig):
        key = self._compute_key(artifact_type, config)

        if key in self.index:
            cache_path = self.cache_dir / artifact_type / f"{key}.pkl"
            if cache_path.exists():
                cache_path.unlink()
            del self.index[key]
            self._save_index()

    def clear(self, artifact_type: Optional[str] = None):
        """Clear cache."""
        if artifact_type:
            cache_type_dir = self.cache_dir / artifact_type
            if cache_type_dir.exists():
                shutil.rmtree(cache_type_dir)
            self.index = {
                k: v for k, v in self.index.items()
                if v.get('artifact_type') != artifact_type
            }
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.index = {}

        self._save_index()

    def get_stats(self) -> Dict[str, Any]:
        total_size = 0
        for key, info in self.index.items():
            cache_path = Path(info.get('path', ''))
            if cache_path.exists():
                total_size += cache_path.stat().st_size

        return {
            'num_items': len(self.index),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / 1024 / 1024,
            'total_size_gb': total_size / 1024 / 1024 / 1024,
        }

class SimplifiedArtifacts:
    """Unified interface for ML artifacts using Hydra + MLflow."""

    def __init__(self, cfg: DictConfig):
        self.config = cfg

        self.rank = dist.get_rank() if dist.is_initialized() else 0

        tracker = ExperimentTracker(cfg)
        self.tracker = RankAwareTracker(tracker, enabled=(self.rank == 0))

        self.cache = SimpleCache(
            cache_dir=cfg.cache_dir,
        )

        self._run_id: Optional[str] = None
        self._current_stage: Optional[str] = None

        logger.debug("SimplifiedArtifacts initialized")

    def start_run(
        self,
        config: DictConfig,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> str | None:

        _run_id = None
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.debug("Starting MLflow run for stage: %s", run_name)
            _run_id = self.tracker.start_run(run_name=run_name, tags=tags, nested=nested)

            flat_cfg = OmegaConf.to_container(config, resolve=True)

            def flatten(d, parent_key="", sep="."):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten(v, new_key).items())
                    else:
                        items.append((new_key, v))
                return dict(items)

            self.tracker.log_params(flatten(flat_cfg))

        if dist.is_initialized():
            dist.barrier()
            _run_id = [_run_id] if _run_id else [None]
            dist.broadcast_object_list(_run_id, src=0)
            self._run_id = str(_run_id[0])
            dist.barrier()

        logger.debug("MLflow run started with ID: %s", self._run_id)
        return self._run_id

    def end_run(self, status: str = "FINISHED"):
        self.tracker.end_run(status=status)

    def log_metrics(
        self,
        metrics: Dict[str, float | str | None],
        stage: Optional[str] = None,
        step: Optional[int] = None
    ):
        prefix = stage or self._current_stage
        self.tracker.log_metrics(metrics, step=step, prefix=prefix)

    def log_artifact(
        self,
        artifact: Any,
        name: str,
        artifact_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ):
        suffix = '.pkl' if artifact_type in ['model', 'dataset'] else '.json'
        try:
            if suffix == '.json':
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix=suffix, delete=False,
                    prefix=f"{artifact_type}_{name}_") as f:
                    json.dump(artifact, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                    temp_path = f.name

                    self.tracker.log_artifact(temp_path, artifact_type)
                    if metadata:
                        self.tracker.log_dict(metadata or {}, f"{artifact_type}_{name}_metadata.json")
            else:
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                ) as f:
                    pickle.dump(artifact, f)
                    f.flush()
                    os.fsync(f.fileno())
                    temp_path = f.name

                    self.tracker.log_artifact(temp_path, artifact_type)
                    if metadata:
                        self.tracker.log_dict(metadata or {}, f"{artifact_type}_{name}_metadata.json")
        except Exception as e:
            logger.error("Failed to log artifact %s of type %s: %s", name, artifact_type, e)
        finally:
            os.unlink(temp_path)

    def get_cached(
        self,
        artifact_type: str,
        config: DictConfig,
        loader: Callable[[], Any],
        suffix: Optional[str] = None
    ) -> tuple[Any, bool]:
        """Get from cache or compute."""
        return self.cache.get(artifact_type, config, suffix, loader)

    def get_run_id(self) -> Optional[str]:
        return self._run_id
