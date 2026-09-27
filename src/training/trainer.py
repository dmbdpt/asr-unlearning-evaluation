import os
import json
import logging
import mlflow
import torch
from torch import optim
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from src.data.datasets.librispeech import collate_fn as collate_fn_espnet
from src.utils.utils import rank0_print

logger = logging.getLogger(__name__)


class StandardTrainer(pl.LightningModule):
    def __init__(self, model, datasets, losses, optimizer_config, batch_size, cfg):
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer_config
        self.batch_size = batch_size
        self.num_workers = cfg["num_workers"]

        self.save_hyperparameters(
            ignore=['model', 'datasets', 'losses', 'optimizer_config'])

        self.train_set = ConcatDataset(
            [datasets['retain'], datasets['forget']])
        self.valid_set = datasets['valid']
        self.test_set = datasets['test']
        self.retain_set = datasets['retain']
        self.forget_set = datasets['forget']

    def train_dataloader(self):
        return DataLoader(
            self.train_set, collate_fn=collate_fn_espnet,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(self.valid_set, collate_fn=collate_fn_espnet,
                          batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(),
                         lr=self.optimizer_config.lr,
                         momentum=self.optimizer_config.momentum)

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss_out = self.model(batch)
        self.log("val/valid_set/loss", loss_out.detach(),
                 prog_bar=True, add_dataloader_idx=False)

    def on_train_end(self):
        """Log training completion tag to MLflow."""
        if mlflow.active_run() and (not dist.is_initialized() or dist.get_rank() == 0):
            mlflow.set_tag("training_status", "completed")


def _freeze_unused_params(model, dataset, collate_fn):
    rank0_print("[Optimization] Checking for unused parameters...")

    loader = DataLoader(dataset, batch_size=2,
                        collate_fn=collate_fn, shuffle=True)
    try:
        batch = next(iter(loader))
    except StopIteration:
        rank0_print(
            "[Optimization] Dataset is empty, skipping parameter check.")
        return

    device = next(model.parameters()).device
    batch_on_device = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_on_device[k] = v.to(device)
        else:
            batch_on_device[k] = v

    model.zero_grad()
    try:
        if hasattr(model, 'original_forward'):
            loss, _, _ = model.original_forward(batch_on_device)
        elif hasattr(model, 'forward'):
            loss = model.forward(batch_on_device)
        else:
            rank0_print("[Optimization] Model has no forward method, skipping parameter check.")
            return
        loss.backward()
    except Exception as e:
        rank0_print(f"[Optimization] Error during parameter check: {e}")
        return

    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            param.requires_grad = False
            unused.append(name)

    model.zero_grad()

    if unused:
        rank0_print(
            f"[Optimization] Frozen {len(unused)} unused parameters: {unused}")
    else:
        rank0_print("[Optimization] No unused parameters found.")


def train_model(model, datasets, clusters=None, config=None, target_cluster_id=None, cfg=None):
    rank0_print("[Training] Starting model training...")
    if datasets is None:
        raise ValueError(
            "Datasets not initialized.")

    train_dataset = datasets["train"]
    if target_cluster_id is not None:
        rank0_print(
            f"[Training] Training only on cluster {target_cluster_id}...")
        if clusters is None:
            raise ValueError(
                "Clusters not found. Run cluster_data() first.")

        train_indices = train_dataset.indices
        new_indices = [
            i for i in train_indices if clusters[i] == target_cluster_id]

        if not new_indices:
            raise ValueError(
                f"No samples found for cluster {target_cluster_id} in training set.")

        rank0_print(
            f"[Training] Filtered training set from {len(train_indices)} to {len(new_indices)} samples.")

        # Ensure strict consistency of indices across ranks for DDP
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            object_list = [new_indices]
            dist.broadcast_object_list(object_list, src=0)
            new_indices = object_list[0]
            dist.barrier()

        train_dataset = Subset(train_dataset.dataset, new_indices)

    datasets_for_training = datasets.copy()
    datasets_for_training["train"] = train_dataset

    train_cfg = config.train_config
    opt_cfg = config.optimizer_config

    trainer_module = StandardTrainer(
        model=model,
        datasets=datasets_for_training,
        losses=config.losses,
        optimizer_config=opt_cfg,
        batch_size=train_cfg.batch_size_train,
        cfg=cfg
    )

    log_dir = train_cfg.log_dir
    os.makedirs(log_dir, exist_ok=True)
    callbacks = []

    if train_cfg.early_stopping.active:
        es_config = train_cfg.early_stopping
        rank0_print(
            f"[Training] Enabling Early Stopping with config: {es_config}")
        callbacks.append(EarlyStopping(
            monitor=es_config.monitor,
            patience=es_config.patience,
            min_delta=es_config.min_delta,
            mode=es_config.mode,
            verbose=True
        ))

    checkpoint_dir = train_cfg.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks.append(ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="finetuned-{epoch:02d}-{val/valid_set/loss:.4f}",
        monitor="val/valid_set/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    ))

    val_check_interval = train_cfg.val_check_interval

    # MLflow settings come from cfg.artifacts, threaded in via the dict built in stage_03_training.py.
    mlflow_experiment_name = cfg["mlflow_experiment_name"]

    if not dist.is_initialized() or dist.get_rank() == 0:
        if mlflow.active_run():
            mlflow.log_params({
                "training.batch_size": train_cfg.batch_size_train,
                "training.max_epochs": train_cfg.max_epochs,
                "training.gradient_clip_val": train_cfg.gradient_clip_val,
                "training.val_check_interval": train_cfg.val_check_interval,
                "training.optimizer.lr": opt_cfg.lr,
                "training.optimizer.momentum": opt_cfg.momentum,
            })
            mlflow.set_tag("pipeline_stage", "finetuning")

    if dist.is_initialized() and dist.get_rank() == 0:
        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow_experiment_name,
            tracking_uri=cfg["tracking_uri"],
            log_model=True,
        )
    else:
        mlflow_logger = None

    trainer = pl.Trainer(
        accelerator=train_cfg.accelerator,
        devices=train_cfg.devices,
        max_epochs=train_cfg.max_epochs,
        logger=mlflow_logger,
        callbacks=callbacks,
        sync_batchnorm=True,
        strategy="auto",
        num_sanity_val_steps=0,
        gradient_clip_val=train_cfg.gradient_clip_val,
        limit_train_batches=train_cfg.limit_train_batches,
        limit_val_batches=train_cfg.limit_val_batches,
        val_check_interval=val_check_interval,
    )

    trainer.fit(trainer_module)
    rank0_print("[Training] Model training complete.")

    if dist.is_available() and dist.is_initialized():
        pass

    if not dist.is_initialized() or dist.get_rank() == 0:
        if mlflow.active_run():
            best_ckpt = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
            last_ckpt = trainer.checkpoint_callback.last_model_path if trainer.checkpoint_callback else None

            if best_ckpt and os.path.exists(best_ckpt):
                mlflow.log_artifact(best_ckpt, artifact_path="checkpoints/finetuned/best")
                mlflow.log_param("training.best_checkpoint", best_ckpt)
                rank0_print(f"[Training] Logged best checkpoint to MLflow: {best_ckpt}")

            if last_ckpt and os.path.exists(last_ckpt):
                mlflow.log_artifact(last_ckpt, artifact_path="checkpoints/finetuned/last")
                rank0_print(f"[Training] Logged last checkpoint to MLflow: {last_ckpt}")

            train_cfg_dump = {
                "train_config": train_cfg,
                "optimizer_config": opt_cfg,
                "model_config": config.model_config,
            }
            with _temp_json_file(train_cfg_dump) as tmp_path:
                mlflow.log_artifact(tmp_path, artifact_path="configs")

            mlflow.set_tag("pipeline_stage", "finetuning_complete")

    return model


def _temp_json_file(data: dict):
    """Context manager that writes data to a temp JSON file and yields its path."""
    import tempfile
    import contextlib
    from omegaconf import DictConfig, OmegaConf

    def convert_dictconfig(d):
        if isinstance(d, DictConfig):
            return OmegaConf.to_container(d, resolve=True)
        if isinstance(d, dict):
            return {k: convert_dictconfig(v) for k, v in d.items()}
        if isinstance(d, list):
            return [convert_dictconfig(item) for item in d]
        return d
    
    data = convert_dictconfig(data)
    
    @contextlib.contextmanager
    def _ctx():
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='_train_cfg.json', delete=False
        ) as f:
            json.dump(data, f, indent=2, default=str)
            tmp_path = f.name
        try:
            yield tmp_path
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return _ctx()


def load_model(model, model_path):
    rank0_print(f"[Model Loading] Loading model state from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path))
        rank0_print("[Model Loading] Model loaded successfully.")
    except Exception as e:
        rank0_print(f"[Model Loading] Could not load model: {e}")
    return model
