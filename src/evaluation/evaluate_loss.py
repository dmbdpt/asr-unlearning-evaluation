from collections import defaultdict
from typing import List

import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from src.data.datasets.librispeech import collate_fn
from src.utils.utils import rank0_print


def _gather_distributed(local_losses: List[float], world_size: int) -> List[float]:
    """All-gather per-rank loss lists and flatten into one list."""
    if not dist.is_initialized():
        return local_losses
    all_ranks: List = [None] * world_size
    dist.all_gather_object(all_ranks, local_losses)
    return [l for rank_l in all_ranks if rank_l for l in rank_l]


def run_evaluate_loss(model, datasets, sets_to_evaluate=["test"], cfg=None):
    if cfg is None:
        cfg = {}
    model.eval()
    all_results = {}

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for set_name in sets_to_evaluate:
        dset = datasets[set_name]

        if dist.is_initialized():
            sampler = DistributedSampler(
                dset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
            )
            desc = f"[RANK: {rank}] Loss Eval: Evaluating {set_name}"
        else:
            sampler = None
            desc = f"[Loss Eval] Evaluating {set_name}"

        loader = DataLoader(
            dset,
            batch_size=cfg.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            sampler=sampler,
            shuffle=False,
        )

        model.to(device)

        # speaker_id -> {"loss_sum": float, "count": int}
        speaker_stats = defaultdict(lambda: {"loss_sum": 0.0, "count": 0})

        for batch in tqdm(loader, desc=desc, position=rank):
            with torch.no_grad():
                batch_on_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                loss = model(batch_on_device)
                spk = batch["speaker_id"]

                if isinstance(loss, torch.Tensor):
                    loss = loss.detach().cpu()

                if loss.ndim == 0:
                    loss = loss.unsqueeze(0)

                for speaker_id, l in zip(spk, loss):
                    speaker_stats[speaker_id]["loss_sum"] += l.item()
                    speaker_stats[speaker_id]["count"] += 1

        if dist.is_initialized():
            all_ranks_stats = [None] * world_size
            dist.all_gather_object(all_ranks_stats, dict(speaker_stats))

            merged_stats = defaultdict(lambda: {"loss_sum": 0.0, "count": 0})
            for rank_stats in all_ranks_stats:
                for spk, stats in rank_stats.items():
                    merged_stats[spk]["loss_sum"] += stats["loss_sum"]
                    merged_stats[spk]["count"] += stats["count"]
            speaker_stats = merged_stats

        organized_res = {
            spk: stats["loss_sum"] / stats["count"] if stats["count"] > 0 else 0.0
            for spk, stats in speaker_stats.items()
        }

        all_results[set_name] = organized_res

    return all_results


def run_evaluate_loss_per_utt(model, datasets, sets_to_evaluate=["test"], cfg=None):
    if cfg is None:
        cfg = {}
    model.eval()
    all_results = {}

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for set_name in sets_to_evaluate:
        dset = datasets[set_name]

        if dist.is_initialized():
            sampler = DistributedSampler(
                dset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            desc = f"[RANK: {rank}] Loss Eval: Evaluating {set_name}"
        else:
            sampler = None
            desc = f"[Loss Eval] Evaluating {set_name}"

        loader = DataLoader(
            dset,
            batch_size=1,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            sampler=sampler,
            shuffle=False,
        )

        # speaker_id -> [loss, loss, ...] (a list, not a dict keyed by utt_id: LibriSpeech's...
        utterance_stats = defaultdict(list)

        for batch in tqdm(loader, desc=desc, position=rank):
            with torch.no_grad():
                batch_on_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                loss = model(batch_on_device)
                spk_ids = batch["speaker_id"]

                if isinstance(loss, torch.Tensor):
                    loss = loss.detach().cpu()

                if loss.ndim == 0:
                    loss = loss.unsqueeze(0)

                for spk_id, l in zip(spk_ids, loss):
                    utterance_stats[spk_id].append(float(l.item()))

        if dist.is_initialized():
            all_ranks_stats = [None] * world_size
            dist.all_gather_object(all_ranks_stats, dict(utterance_stats))

            merged_stats = defaultdict(list)
            for rank_stats in all_ranks_stats:
                for spk_id, utt_losses in rank_stats.items():
                    merged_stats[spk_id].extend(utt_losses)

            utterance_stats = merged_stats

        all_results[set_name] = dict(utterance_stats)

    return all_results


def compute_emd_forget_test(model, datasets, cfg=None):
    """EMD between the forget-set and test-set loss distributions."""
    if cfg is None:
        cfg = {}

    from scipy.stats import wasserstein_distance

    model.eval()

    if "forget" not in datasets:
        rank0_print(
            "[EMD] Warning: 'forget' dataset not found, skipping EMD calculation"
        )
        return {"emd_forget_test": None}
    if "test" not in datasets:
        rank0_print("[EMD] Warning: 'test' dataset not found, skipping EMD calculation")
        return {"emd_forget_test": None}

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = (
        torch.device(f"cuda:{rank}")
        if dist.is_initialized()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    forget_losses = _collect_losses_for_emd(
        model, datasets["forget"], device, cfg, rank, world_size
    )

    test_losses = _collect_losses_for_emd(
        model, datasets["test"], device, cfg, rank, world_size
    )

    forget_losses = _gather_distributed(forget_losses, world_size)
    test_losses = _gather_distributed(test_losses, world_size)

    forget_losses = np.array(forget_losses)
    test_losses = np.array(test_losses)

    if len(forget_losses) < 2 or len(test_losses) < 2:
        rank0_print("[EMD] Warning: Not enough samples to compute EMD")
        return {"emd_forget_test": None}

    emd_value = wasserstein_distance(forget_losses, test_losses)

    rank0_print(
        f"[EMD] Forget set losses: mean={forget_losses.mean():.4f}, std={forget_losses.std():.4f}, n={len(forget_losses)}"
    )
    rank0_print(
        f"[EMD] Test set losses: mean={test_losses.mean():.4f}, std={test_losses.std():.4f}, n={len(test_losses)}"
    )
    rank0_print(f"[EMD] EMD (forget vs test): {emd_value:.4f}")

    return {"emd_forget_test": float(emd_value)}


def compute_emd_forget_post_vs_test_pre(
    unlearned_model, pre_unlearned_model, datasets, cfg=None
):
    """Compute Earth Mover's Distance (EMD) between: 1."""
    if cfg is None:
        cfg = {}

    from scipy.stats import wasserstein_distance

    unlearned_model.eval()
    pre_unlearned_model.eval()

    if "forget" not in datasets:
        rank0_print(
            "[EMD] Warning: 'forget' dataset not found, skipping EMD calculation"
        )
        return {"emd_forget_post_test_pre": None}
    if "test" not in datasets:
        rank0_print("[EMD] Warning: 'test' dataset not found, skipping EMD calculation")
        return {"emd_forget_post_test_pre": None}

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = (
        torch.device(f"cuda:{rank}")
        if dist.is_initialized()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    forget_losses = _collect_losses_for_emd(
        unlearned_model, datasets["forget"], device, cfg, rank, world_size
    )

    test_losses = _collect_losses_for_emd(
        pre_unlearned_model, datasets["test"], device, cfg, rank, world_size
    )

    forget_losses = _gather_distributed(forget_losses, world_size)
    test_losses = _gather_distributed(test_losses, world_size)

    forget_losses = np.array(forget_losses)
    test_losses = np.array(test_losses)

    if len(forget_losses) < 2 or len(test_losses) < 2:
        rank0_print("[EMD] Warning: Not enough samples to compute EMD")
        return {"emd_forget_post_test_pre": None}

    emd_value = wasserstein_distance(forget_losses, test_losses)

    rank0_print(
        f"[EMD] Forget (post-unlearn) losses: mean={forget_losses.mean():.4f}, std={forget_losses.std():.4f}, n={len(forget_losses)}"
    )
    rank0_print(
        f"[EMD] Test (pre-unlearn) losses: mean={test_losses.mean():.4f}, std={test_losses.std():.4f}, n={len(test_losses)}"
    )
    rank0_print(f"[EMD] EMD (forget post vs test pre): {emd_value:.4f}")

    return {"emd_forget_post_test_pre": float(emd_value)}


def compute_emd_forget_post_vs_test_pre_computed(
    unlearned_model, test_losses_precomputed, datasets, cfg=None
):
    """Compute Earth Mover's Distance (EMD) between: 1."""
    if cfg is None:
        cfg = {}

    from scipy.stats import wasserstein_distance

    unlearned_model.eval()

    if "forget" not in datasets:
        rank0_print(
            "[EMD] Warning: 'forget' dataset not found, skipping EMD calculation"
        )
        return {"emd_forget_post_test_pre": None}

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = (
        torch.device(f"cuda:{rank}")
        if dist.is_initialized()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    forget_losses = _collect_losses_for_emd(
        unlearned_model, datasets["forget"], device, cfg, rank, world_size
    )

    forget_losses = _gather_distributed(forget_losses, world_size)
    forget_losses = np.array(forget_losses)

    if len(forget_losses) < 2 or len(test_losses_precomputed) < 2:
        rank0_print("[EMD] Warning: Not enough samples to compute EMD")
        return {"emd_forget_post_test_pre": None}

    emd_value = wasserstein_distance(forget_losses, test_losses_precomputed)

    rank0_print(
        f"[EMD] Forget (post-unlearn) losses: mean={forget_losses.mean():.4f}, std={forget_losses.std():.4f}, n={len(forget_losses)}"
    )
    rank0_print(
        f"[EMD] Test (pre-unlearn) losses: mean={test_losses_precomputed.mean():.4f}, std={test_losses_precomputed.std():.4f}, n={len(test_losses_precomputed)}"
    )
    rank0_print(f"[EMD] EMD (forget post vs test pre): {emd_value:.4f}")

    return {"emd_forget_post_test_pre": float(emd_value)}

def compute_emd_ref_loss_vs_dataset(
    unlearned_model, test_losses_precomputed, dataset, cfg=None
):
    """Compute Earth Mover's Distance (EMD) between: 1."""
    if cfg is None:
        cfg = {}

    from scipy.stats import wasserstein_distance

    unlearned_model.eval()

    if dataset is None:
        rank0_print(
            "[EMD] Warning: Dataset not specified, skipping EMD calculation"
        )
        return {"emd_reference_post_test_pre": None}

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = (
        torch.device(f"cuda:{rank}")
        if dist.is_initialized()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    forget_losses = _collect_losses_for_emd(
        unlearned_model, dataset, device, cfg, rank, world_size
    )

    forget_losses = _gather_distributed(forget_losses, world_size)
    forget_losses = np.array(forget_losses)

    if len(forget_losses) < 2 or len(test_losses_precomputed) < 2:
        rank0_print("[EMD] Warning: Not enough samples to compute EMD")
        return {"emd_reference_post_test_pre": None}

    emd_value = wasserstein_distance(forget_losses, test_losses_precomputed)

    rank0_print(
        f"[EMD] Forget (post-unlearn) losses: mean={forget_losses.mean():.4f}, std={forget_losses.std():.4f}, n={len(forget_losses)}"
    )
    rank0_print(
        f"[EMD] Test (pre-unlearn) losses: mean={test_losses_precomputed.mean():.4f}, std={test_losses_precomputed.std():.4f}, n={len(test_losses_precomputed)}"
    )
    rank0_print(f"[EMD] EMD (forget post vs test pre): {emd_value:.4f}")

    return {"emd_reference_post_test_pre": float(emd_value)}


def compute_test_losses_pre_unlearning(model, datasets, cfg=None):
    """Compute and return test losses before unlearning. This can be cached and reused."""
    if cfg is None:
        cfg = {}

    model.eval()

    if "test" not in datasets:
        rank0_print("[EMD] Warning: 'test' dataset not found, returning empty array")
        return np.array([])

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = (
        torch.device(f"cuda:{rank}")
        if dist.is_initialized()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    test_losses = _collect_losses_for_emd(
        model, datasets["test"], device, cfg, rank, world_size
    )

    test_losses = _gather_distributed(test_losses, world_size)
    return np.array(test_losses)


def _collect_losses_for_emd(model, dataset, device, cfg, rank, world_size):
    """
    Helper function to collect losses from a dataset for EMD calculation.
    """
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        desc = f"[RANK: {rank}] EMD Loss: Collecting losses"
    else:
        sampler = None
        desc = f"[EMD Loss] Collecting losses"

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        sampler=sampler,
        shuffle=False,
    )

    model.to(device)
    losses = []

    for batch in tqdm(loader, desc=desc, position=rank):
        with torch.no_grad():
            batch_on_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            loss = model(batch_on_device)

            if isinstance(loss, torch.Tensor):
                loss = loss.detach().cpu()

            if loss.ndim == 0:
                loss = loss.unsqueeze(0)

            losses.extend(loss.numpy().tolist())

    return losses
