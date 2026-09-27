#!/usr/bin/env python3
"""Script to evaluate utterance-level loss for a given experiment and run."""

import argparse
import mlflow
import os
import torch
from omegaconf import DictConfig, OmegaConf
import warnings
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.data.datasets.librispeech import collate_fn
warnings.filterwarnings("ignore")

from src.data.data_handler import DataHandler
from src.pipeline.stage_00_initialization import run_initialization
from src.pipeline.stage_01_data_preparation import run_load_datasets
from src.models.espnet import ESPnetASRWrapper
from src.utils.utils import rank0_print


def evaluate_loss_utt_level_with_device(model, datasets, sets_to_evaluate=["test"], cfg=None, device=None):
    """
    Evaluate utterance-level loss with explicit device handling.
    """
    if cfg is None:
        cfg = {}
    model.eval()
    all_results = {}

    for set_name in sets_to_evaluate:
        dset = datasets[set_name]

        loader = DataLoader(
            dset,
            batch_size=cfg.get("batch_size", 1),
            collate_fn=collate_fn,
            num_workers=cfg.get("num_workers", 0),
            shuffle=False
        )

        utt_losses = []

        for batch in tqdm(loader, desc=f"Evaluating {set_name}"):
            with torch.no_grad():
                batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                loss = model(batch_on_device)
                utt_losses.extend(loss.detach().cpu().tolist())

        all_results[set_name] = utt_losses

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate utterance-level loss from MLflow run')
    parser.add_argument('--experiment_id', type=str, required=True, help='MLflow Experiment ID')
    parser.add_argument('--run_id', type=str, required=True, help='MLflow Run ID')
    parser.add_argument('--splits', nargs='+', default=['retain','forget', 'test'], help='Data splits to evaluate (default: train test)')
    parser.add_argument('--tracking_uri', type=str, default='sqlite:///mlflow.db', help='MLflow tracking URI')
    parser.add_argument('--config_path', type=str, default='config/config.yaml', help='Path to base config file')
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    rank0_print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    try:
        run = mlflow.get_run(run_id=args.run_id)
        rank0_print(f"Retrieved run: {run.info.run_id}")
    except Exception as e:
        raise ValueError(f"Could not retrieve run {args.run_id}: {e}")

    if run.info.experiment_id != args.experiment_id:
        rank0_print(f"Warning: Run experiment ID {run.info.experiment_id} does not match provided {args.experiment_id}")

    config = None
    try:
        config_path = mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path="config.yaml")
        config = OmegaConf.load(config_path)
        rank0_print("Loaded config from run artifact 'config.yaml'")
    except Exception as e:
        rank0_print(f"Could not download config.yaml artifact: {e}")
        # Fallback to base config file and override with run params
        rank0_print(f"Loading base config from {args.config_path}")
        config = OmegaConf.load(args.config_path)
        params = run.data.params
        if params:
            # Convert params to nested dict for OmegaConf
            def _parse_param_value(v):
                # MLflow params are strings, but we try to convert to appropriate type
                if isinstance(v, bool):
                    return v
                if isinstance(v, int):
                    return v
                if isinstance(v, float):
                    return v
                v_lower = v.lower()
                if v_lower == 'true':
                    return True
                if v_lower == 'false':
                    return False
                try:
                    return int(v)
                except ValueError:
                    pass
                try:
                    return float(v)
                except ValueError:
                    pass
                return v

            for key, value in params.items():
                parsed_value = _parse_param_value(value)
                try:
                    OmegaConf.update(config, key, parsed_value)
                except Exception as e:
                    rank0_print(f"Warning: Could not set config key {key} with value {parsed_value}: {e}")
            rank0_print("Updated base config with run parameters")
        else:
            rank0_print("No parameters found in run, using base config only")

    rank0_print("Initializing artifacts...")
    artifacts = run_initialization(config)
    rank0_print("Loading datasets...")
    data_handler = run_load_datasets(config, artifacts)
    datasets = data_handler.datasets
    
    # Create forget split containing utterances with speaker_id "163" from the train split
    if "train" in datasets:
        train_set = datasets["train"]
        forget_indices = []
        try:
            if hasattr(train_set, 'speaker_ids'):
                speaker_ids = train_set.speaker_ids
            elif hasattr(train_set, 'dataset') and hasattr(train_set.dataset, 'speaker_ids'):
                speaker_ids = train_set.dataset.speaker_ids
            else:
                # Fallback: we'll need to iterate through the dataset
                speaker_ids = []
                for i in range(len(train_set)):
                    item = train_set[i]
                    speaker_ids.append(item.get("speaker_id", ""))
            
            for i, spk_id in enumerate(speaker_ids):
                if str(spk_id) == "163":
                    forget_indices.append(i)
        except Exception as e:
            rank0_print(f"Warning: Could not pre-compute forget indices for train: {e}")
            forget_indices = []  # We'll try to create the forget dataset by iterating in the DataLoader later? But we want to create a Subset now.
        
        if forget_indices:
            from torch.utils.data import Subset
            datasets["forget"] = Subset(train_set, forget_indices)
            rank0_print(f"Created forget split with {len(forget_indices)} utterances (speaker_id='163') from train set")
            
            if "forget" not in args.splits:
                args.splits = list(args.splits) + ["forget"]
        else:
            rank0_print("Warning: No utterances found with speaker_id='163' in train set")
    else:
        rank0_print("Warning: No train set found in datasets, cannot create forget split")
    
    splits_to_evaluate = []
    for split in args.splits:
        if split in datasets:
            splits_to_evaluate.append(split)
        else:
            rank0_print(f"Warning: Split '{split}' not found in datasets, skipping.")
    
    if not splits_to_evaluate:
        raise ValueError("No valid splits to evaluate after checking dataset availability.")
    
    rank0_print(f"Evaluating loss for splits: {splits_to_evaluate}")

    rank0_print("Downloading model artifact...")
    model_artifact_paths = [
        # "model", "checkpoints/finetuned_model.pt", "finetuned_model.pt", "model.pth"...
        "last/last.ckpt",
    ]
    model_path = None
    for artifact_path in model_artifact_paths:
        try:
            model_path = mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path=artifact_path)
            rank0_print(f"Found model artifact at: {artifact_path}")
            break
        except Exception as e:
            rank0_print(f"Artifact {artifact_path} not found: {e}")
            continue

    if model_path is None:
        raise FileNotFoundError("Could not find model artifact in run. Tried paths: " + ", ".join(model_artifact_paths))

    rank0_print(f"Loading model from {model_path}")
    model_tag = config.training.model_tag
    model_config = config.training.model_config
    model = ESPnetASRWrapper.from_pretrained(model_tag, **model_config)
    checkpoint = torch.load(model_path, map_location='cpu')
    # Extract state_dict from checkpoint (handle PyTorch Lightning checkpoints)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if any(key.startswith('model.') for key in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[6:]  # remove 'model.'
            new_state_dict[name] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    rank0_print(f"Model loaded successfully and moved to {device}")

    rank0_print(f"Evaluating loss for splits: {splits_to_evaluate}")
    # Use custom evaluation function that handles device placement
    all_results = evaluate_loss_utt_level_with_device(
        model=model,
        datasets=datasets,
        sets_to_evaluate=splits_to_evaluate,
        cfg={
            "batch_size": config.evaluation.batch_size,
            "num_workers": config.evaluation.num_workers
        },
        device=next(model.parameters()).device  # Get the device the model is on
    )

    print("\n=== Utterance-Level Loss Results ===")
    for set_name, losses in all_results.items():
        print(f"\nSet: {set_name}")
        print(f"Number of utterances: {len(losses)}")
        print(f"Loss values (first 5): {losses[:5]}")
        if len(losses) > 5:
            print(f"Loss values (last 5): {losses[-5:]}")
        print(f"Mean loss: {sum(losses)/len(losses):.6f}")
        print(f"Std loss: {torch.tensor(losses).std().item():.6f}")

    output_file = f"utt_loss_exp{args.experiment_id}_run{args.run_id}.pt"
    torch.save(all_results, output_file)
    rank0_print(f"Saved full results to {output_file}")

    import matplotlib.pyplot as plt
    import numpy as np

    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
    plt.figure(figsize=(10, 6))
    for i, (set_name, losses) in enumerate(all_results.items()):
        if len(losses) == 0:
            continue
        # Plot histogram with density=True to get PDF
        plt.hist(losses, bins=50, density=True, alpha=0.5, color=colors[i % len(colors)], label=set_name)
    plt.xlabel("Loss")
    plt.ylabel("Density")
    plt.title(f"Utterance-Level Loss Distribution for Experiment {args.experiment_id} Run {args.run_id}")
    plt.legend()
    plot_file = f"utt_loss_distribution_exp{args.experiment_id}_run{args.run_id}.png"
    plt.savefig(plot_file)
    rank0_print(f"Saved loss distribution plot to {plot_file}")

if __name__ == "__main__":
    main()