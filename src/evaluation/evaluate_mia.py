import os
import re
import zlib
import tempfile
import shutil
import traceback
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Subset, random_split
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from src.membership_inference.modules.speaker_level_attacker import SpeakerLevelAttacker
from src.membership_inference.modules.abs_attacker import Attacker
from src.membership_inference.modules.loss_feature_extractor import LossFeatureExtractorEspnetASR
from src.utils.utils import rank0_print

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _barrier_if_distributed():
    if dist.is_initialized():
        dist.barrier()


def _cleanup_temp_dir(temp_dir: str, artifact_dir: Optional[str]) -> None:
    if not artifact_dir and temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def _assert_base_dataset(name: str, dataset) -> None:
    if dataset is None:
        raise ValueError(f"Expected datasets['{name}'] to exist")
    if isinstance(dataset, Subset):
        raise ValueError(f"Expected datasets['{name}'] to be a base/raw dataset, but got Subset")


def _dataset_get_metadata(dataset, idx: int, dicted: bool = True):
    """Supports: - raw/base datasets - Subset objects wrapping raw datasets"""
    if isinstance(dataset, Subset):
        base_idx = dataset.indices[idx]
        return _dataset_get_metadata(dataset.dataset, base_idx, dicted=dicted)

    if hasattr(dataset, "get_metadata"):
        try:
            return dataset.get_metadata(idx, dicted=dicted)
        except TypeError:
            return dataset.get_metadata(idx)

    raise AttributeError(
        f"Dataset of type {type(dataset).__name__} does not implement get_metadata(...)"
    )


def _speaker_id_from_metadata(item) -> str:
    if not isinstance(item, dict):
        raise ValueError("Expected dataset metadata to be a dict")

    spk = item.get("speaker_id", item.get("spk_id", None))
    if spk is not None:
        return str(spk)

    raise ValueError("Expected metadata dict with 'speaker_id' or 'spk_id'")


# LibriSpeech corpus directories, longest first so train-clean-100 wins over a...
_CORPUS_RE = re.compile(
    r"(train-clean-360|train-clean-100|train-other-500|"
    r"dev-clean|dev-other|test-clean|test-other)"
)


def _corpus_from_metadata(item) -> str:
    """Which LibriSpeech corpus an utterance came from ("test-clean", "test-other", ...)."""
    if isinstance(item, dict):
        for key in ("split", "corpus", "subset"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
        for key in ("wav_path", "wav"):
            value = item.get(key)
            if isinstance(value, str):
                match = _CORPUS_RE.search(value)
                if match:
                    return match.group(1)
    return "unknown"


def _speaker_ids_in_dataset(dataset) -> set[str]:
    return {
        _speaker_id_from_metadata(_dataset_get_metadata(dataset, i, dicted=True))
        for i in tqdm(
            range(len(dataset)),
            desc="Extracting speaker IDs from dataset",
            disable=(_get_rank() != 0),
        )
    }


def _build_feature_extractor(model, cfg: Dict[str, Any], feature_dir: str, name: str):
    params = {
        "feature_extractor_name": name,
        "save_features_folder": cfg.cache_dir if cfg.cache_dir is not None else feature_dir,
        "cache_save_frequency": cfg.cache_save_frequency,
        "loss_att": cfg.loss_att,
        "loss_ctc": cfg.loss_ctc,
        "loss_cer": cfg.loss_cer,
    }
    return LossFeatureExtractorEspnetASR(model_to_attack=model, cfg=params)


def _build_classifier(cfg: Dict[str, Any], seed: int):
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state if cfg.random_state is not None else seed,
        n_jobs=cfg.n_jobs,
    )


def _move_model_to_device(model) -> None:
    if torch.cuda.is_available():
        rank = _get_rank()
        if dist.is_initialized() and torch.cuda.device_count() > 0:
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if hasattr(model, "to"):
        model.to(device)


def _resolve_forget_and_retain_indices(
    datasets: Dict[str, Any],
    forget_speakers: Optional[Sequence[Any]] = None,
) -> Tuple[List[int], List[int]]:
    """Priority: 1) explicit datasets['forget_idx'] / datasets['retain_idx'] 2) derive from..."""
    train_raw = datasets.get("train_raw", None)
    _assert_base_dataset("train_raw", train_raw)

    forget_idx = list(datasets.get("forget_idx", []))
    retain_idx = list(datasets.get("retain_idx", []))

    if forget_idx and retain_idx:
        return forget_idx, retain_idx

    if forget_speakers is None:
        retain_idx = list(range(len(train_raw)))
        return forget_idx, retain_idx

    forget_set = {str(s) for s in forget_speakers}
    forget_idx = []
    retain_idx = []

    for abs_i in range(len(train_raw)):
        spk = _speaker_id_from_metadata(train_raw.get_metadata(abs_i, dicted=True))
        (forget_idx if spk in forget_set else retain_idx).append(abs_i)

    return forget_idx, retain_idx


NONMEMBER_SPLIT_MODES = ("speaker", "utterance")


def _split_test_nonmembers(
    test_dataset,
    seed: int = 42,
    train_split_ratio: float = 0.5,
    split_by: str = "speaker",
):
    """Split the non-member pool into the attacker's training half and its eval half."""
    if split_by not in NONMEMBER_SPLIT_MODES:
        raise ValueError(
            f"split_by must be one of {NONMEMBER_SPLIT_MODES}, got {split_by!r}"
        )

    n_total = len(test_dataset)
    n_test_train = int(n_total * train_split_ratio)
    if n_test_train < 0 or n_test_train > n_total:
        raise ValueError("Invalid train_split_ratio for test split")

    if split_by == "utterance":
        return random_split(
            test_dataset,
            [n_test_train, n_total - n_test_train],
            generator=torch.Generator().manual_seed(seed),
        )

    # speaker -> indices, grouped by corpus
    by_corpus: Dict[str, Dict[str, List[int]]] = {}
    for i in range(n_total):
        meta = _dataset_get_metadata(test_dataset, i, dicted=True)
        corpus = _corpus_from_metadata(meta)
        speaker = _speaker_id_from_metadata(meta)
        by_corpus.setdefault(corpus, {}).setdefault(speaker, []).append(i)

    train_idx: List[int] = []
    eval_idx: List[int] = []
    for corpus in sorted(by_corpus):
        speakers = by_corpus[corpus]
        # Seed per corpus so that adding or removing one corpus does not reshuffle the other, and...
        g = torch.Generator().manual_seed(
            seed + int(zlib.crc32(corpus.encode("utf-8")) % 10_000)
        )
        order = torch.randperm(len(speakers), generator=g).tolist()
        names = sorted(speakers)
        n_corpus = sum(len(speakers[n]) for n in names)
        target = n_corpus * train_split_ratio

        taken = 0
        n_train_spk = 0
        for j in order:
            rows = speakers[names[j]]
            if taken < target:
                train_idx.extend(rows)
                taken += len(rows)
                n_train_spk += 1
            else:
                eval_idx.extend(rows)

        rank0_print(
            f"[MIA] non-member split ({corpus}): "
            f"{n_train_spk}/{len(names)} speakers -> attacker-train "
            f"({taken}/{n_corpus} utterances)"
        )

    if not train_idx or not eval_idx:
        raise ValueError(
            "Speaker-disjoint non-member split left one half empty; the test pool "
            "needs at least two speakers per corpus (use split_by='utterance' only "
            "for the ablation)"
        )

    overlap = {
        _speaker_id_from_metadata(_dataset_get_metadata(test_dataset, i, dicted=True))
        for i in train_idx
    } & {
        _speaker_id_from_metadata(_dataset_get_metadata(test_dataset, i, dicted=True))
        for i in eval_idx
    }
    if overlap:
        raise AssertionError(f"Non-member split is not speaker-disjoint: {sorted(overlap)}")

    return Subset(test_dataset, sorted(train_idx)), Subset(test_dataset, sorted(eval_idx))


def _prepare_forget_eval_sets(
    datasets: Dict[str, Any],
    forget_speakers: Optional[Sequence[Any]],
    seed: int,
    train_split_ratio: float,
    nonmember_split_by: str = "speaker",
) -> Dict[str, Any]:
    train_raw = datasets.get("train_raw", None)
    test_subset = datasets.get("test", datasets.get("test_raw", None))

    _assert_base_dataset("train_raw", train_raw)
    if test_subset is None:
        raise ValueError("Expected datasets['test'] or datasets['test_raw']")

    forget_idx, retain_idx = _resolve_forget_and_retain_indices(
        datasets=datasets,
        forget_speakers=forget_speakers,
    )

    if len(forget_idx) == 0:
        raise ValueError("Forget set is empty")
    if len(retain_idx) == 0:
        raise ValueError("Retain set is empty")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(retain_idx), generator=g).tolist()
    retain_shuf = [retain_idx[i] for i in perm]

    if len(test_subset) >= len(retain_idx):
        n_retain_train = int(len(retain_idx) * train_split_ratio)
    else:
        n_retain_train = int(len(test_subset) * train_split_ratio)

    if n_retain_train <= 0:
        raise ValueError("No retained training members left after split")
    if n_retain_train > len(retain_shuf):
        raise ValueError("Invalid retained-member split")

    retain_train_idx = retain_shuf[:n_retain_train]
    forget_eval_idx = forget_idx if forget_idx else retain_shuf[
                                               n_retain_train:2*n_retain_train]

    retain_train_ds = Subset(train_raw, retain_train_idx)
    forget_eval_ds = Subset(train_raw, forget_eval_idx)


    test_train_ds, test_eval_ds = _split_test_nonmembers(
        test_dataset=test_subset,
        seed=seed,
        train_split_ratio=train_split_ratio,
        split_by=nonmember_split_by,
    )

    return {
        "retain_train_ds": retain_train_ds,
        "forget_eval_ds": forget_eval_ds,
        "test_train_ds": test_train_ds,
        "test_eval_ds": test_eval_ds,
        "forget_idx": forget_idx,
        "retain_idx": retain_idx,
    }


# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------

def _row_from_metadata(
    metadata: Dict[str, Any],
    utt_in_set: int,
    spk_in_set: int,
    fallback_id: str,
) -> Dict[str, Any]:
    return {
        "wav": metadata.get("wav_path", metadata.get("wav", "")),
        "wrd": metadata.get("wrd", metadata.get("text", "")),
        "ID": (
            # Prefer wav path — always globally unique
            metadata.get("wav_path", metadata.get("wav", ""))
            or (
                # Fall back to composite LibriSpeech-style key
                "{}-{}-{}".format(
                    metadata.get("speaker_id", metadata.get("spk_id", "")),
                    metadata.get("chapt_id", metadata.get("chapter_id", "")),
                    metadata.get("utt_id", ""),
                )
            )
            or fallback_id
        ),
        "dataset": metadata.get("dataset", "librispeech"),
        "language": metadata.get("language", "en"),
        "utt_in_set": int(utt_in_set),
        "spk_in_set": int(spk_in_set),
        "speaker_id": metadata.get("speaker_id", metadata.get("spk_id", "unknown")),
        "duration": metadata.get("wav_lens", metadata.get("duration", None)),
        # Carried so metrics can be reported per LibriSpeech corpus; ignored by...
        "corpus": _corpus_from_metadata(metadata),
    }


def create_mi_csv_from_dataset(
    dataset,
    output_path: str,
    utt_label: int,
    spk_label: Optional[int] = None,
) -> str:
    if spk_label is None:
        spk_label = utt_label

    ls_root = ''
    tmp = dataset
    while isinstance(tmp, Subset):
        tmp = tmp.dataset
    if hasattr(tmp, "root"):
        ls_root = tmp.root

    rows = []
    for i in range(len(dataset)):
        metadata = _dataset_get_metadata(dataset, i, dicted=True)
        if ls_root and "wav" in metadata and not os.path.isabs(metadata["wav"]):
            metadata["wav"] = os.path.join(ls_root, 'LibriSpeech', metadata["wav"])
        rows.append(
            _row_from_metadata(
                metadata=metadata,
                utt_in_set=utt_label,
                spk_in_set=spk_label,
                fallback_id=f"idx_{i}",
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return output_path


def _concat_csvs(csv_paths: List[str], output_path: str) -> str:
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(output_path, index=False)
    return output_path


def _prepare_forget_eval_csvs(
    datasets: Dict[str, Any],
    temp_dir: str,
    forget_speakers: Optional[Sequence[Any]],
    seed: int,
    train_split_ratio: float,
    nonmember_split_by: str = "speaker",
) -> Dict[str, Any]:
    mia_sets = _prepare_forget_eval_sets(
        datasets=datasets,
        forget_speakers=forget_speakers,
        seed=seed,
        train_split_ratio=train_split_ratio,
        nonmember_split_by=nonmember_split_by,
    )

    retain_train_ds = mia_sets["retain_train_ds"]
    forget_eval_ds = mia_sets["forget_eval_ds"]
    test_train_ds = mia_sets["test_train_ds"]
    test_eval_ds = mia_sets["test_eval_ds"]

    retain_train_csv = create_mi_csv_from_dataset(
        retain_train_ds,
        os.path.join(temp_dir, "retain_train.csv"),
        utt_label=1,
        spk_label=1,
    )
    test_train_csv = create_mi_csv_from_dataset(
        test_train_ds,
        os.path.join(temp_dir, "test_train.csv"),
        utt_label=0,
        spk_label=0,
    )
    forget_eval_csv = create_mi_csv_from_dataset(
        forget_eval_ds,
        os.path.join(temp_dir, "forget_eval.csv"),
        utt_label=1,
        spk_label=1,
    )
    test_eval_csv = create_mi_csv_from_dataset(
        test_eval_ds,
        os.path.join(temp_dir, "test_eval.csv"),
        utt_label=0,
        spk_label=0,
    )

    train_csv = _concat_csvs(
        [retain_train_csv, test_train_csv],
        os.path.join(temp_dir, "mia_train.csv"),
    )
    eval_csv = _concat_csvs(
        [forget_eval_csv, test_eval_csv],
        os.path.join(temp_dir, "mia_eval.csv"),
    )

    return {
        "retain_train_ds": retain_train_ds,
        "forget_eval_ds": forget_eval_ds,
        "test_train_ds": test_train_ds,
        "test_eval_ds": test_eval_ds,
        "retain_train_csv": retain_train_csv,
        "test_train_csv": test_train_csv,
        "forget_eval_csv": forget_eval_csv,
        "test_eval_csv": test_eval_csv,
        "train_csv": train_csv,
        "eval_csv": eval_csv,
        "forget_idx": mia_sets["forget_idx"],
        "retain_idx": mia_sets["retain_idx"],
    }


# ---------------------------------------------------------------------
# 1) Forget-specific utterance-level MIA
# ---------------------------------------------------------------------

def evaluate_mia_forget_utt_level(
    model,
    datasets: Dict[str, Any],
    forget_speakers: Optional[Sequence[Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    artifact_dir: Optional[str] = None,
    seed: int = 42,
    stage: str = "post",
) -> Dict[str, float]:
    if cfg is None:
        cfg = {}

    rank = _get_rank()
    if dist.is_initialized() and rank != 0:
        rank0_print("[MIA-Utt] Running only on rank 0, returning empty results for other ranks")
        return {}

    temp_dir = (
        os.path.join(artifact_dir, "mia_utt_temp")
        if artifact_dir
        else tempfile.mkdtemp(prefix="mia_utt_")
    )
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        rank0_print("[MIA-Utt] Preparing forget-specific utterance-level MIA...")

        csv_sets = _prepare_forget_eval_csvs(
            datasets=datasets,
            temp_dir=temp_dir,
            forget_speakers=forget_speakers,
            seed=seed,
            train_split_ratio=cfg.mia_train_split_ratio,
        )

        rank0_print("[MIA-Utt] Dataset splits:")
        rank0_print(f"  Train members (retained): {len(csv_sets['retain_train_ds'])}")
        rank0_print(f"  Train non-members (test): {len(csv_sets['test_train_ds'])}")
        rank0_print(f"  Eval members (forgotten): {len(csv_sets['forget_eval_ds'])}")
        rank0_print(f"  Eval non-members (test): {len(csv_sets['test_eval_ds'])}")
        rank0_print(f"  Train CSV: {csv_sets['train_csv']}")
        rank0_print(f"  Eval CSV: {csv_sets['eval_csv']}")

        _move_model_to_device(model)
        model.eval()

        loss_features = _build_feature_extractor(
            model=model,
            cfg=cfg,
            feature_dir=os.path.join(temp_dir, "features"),
            name=f"loss_extractor_forget_utt_{stage}",
        )

        classifier = _build_classifier(cfg=cfg, seed=seed)

        mia_cfg = {
            "datasets": {
                "train": csv_sets["train_csv"],
                "dev": csv_sets["eval_csv"],
                "test": csv_sets["eval_csv"],
            },
            "batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers,
            "class_label": "utt_in_set",
            "label": "utt_in_set",
            "classifier": classifier,
            "feature_extractors": [loss_features],
        }

        rank0_print("[MIA-Utt] Creating attacker...")
        attacker = Attacker(mia_cfg)

        override = cfg.override

        rank0_print("[MIA-Utt] Training classifier...")
        attacker.train_classifier(override=override)

        rank0_print("[MIA-Utt] Evaluating classifier...")
        results = attacker.evaluate(split="test", override=override)

        rank0_print("[MIA-Utt] Results:")
        for k, v in results.items():
            rank0_print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")

        return results

    except Exception as e:
        rank0_print(f"[MIA-Utt] Error: {e}")
        traceback.print_exc()
        return {}

    finally:
        _cleanup_temp_dir(temp_dir, artifact_dir)


# ---------------------------------------------------------------------
# 2) Forget-specific speaker-level MIA
# ---------------------------------------------------------------------

def evaluate_mia_forget_speaker_level(
    model,
    datasets: Dict[str, Any],
    forget_speakers: Optional[Sequence[Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    artifact_dir: Optional[str] = None,
    seed: int = 42,
    stage: str = "post",
) -> Dict[str, float]:
    if cfg is None:
        cfg = {}

    rank = _get_rank()
    if dist.is_initialized() and rank != 0:
        rank0_print("[MIA-Spk] Running only on rank 0, returning empty results for other ranks")
        return {}

    temp_dir = (
        os.path.join(artifact_dir, "mia_spk_temp")
        if artifact_dir
        else tempfile.mkdtemp(prefix="mia_spk_")
    )
    os.makedirs(temp_dir, exist_ok=True)

    try:
        rank0_print("[MIA-Spk] Preparing forget-specific speaker-level MIA...")

        csv_sets = _prepare_forget_eval_csvs(
            datasets=datasets,
            temp_dir=temp_dir,
            forget_speakers=forget_speakers,
            seed=seed,
            train_split_ratio=cfg.mia_train_split_ratio,
        )

        rank0_print("[MIA-Spk] Dataset splits:")
        rank0_print(f"  Train members (retained): {len(csv_sets['retain_train_ds'])}")
        rank0_print(f"  Train non-members (test): {len(csv_sets['test_train_ds'])}")
        rank0_print(f"  Eval members (forgotten): {len(csv_sets['forget_eval_ds'])}")
        rank0_print(f"  Eval non-members (test): {len(csv_sets['test_eval_ds'])}")
        rank0_print(f"  Train CSV: {csv_sets['train_csv']}")
        rank0_print(f"  Eval CSV: {csv_sets['eval_csv']}")

        _move_model_to_device(model)
        model.eval()

        loss_features = _build_feature_extractor(
            model=model,
            cfg=cfg,
            feature_dir=os.path.join(temp_dir, "features"),
            name=f"loss_extractor_forget_spk_{stage}",
        )

        classifier = _build_classifier(cfg=cfg, seed=seed)

        attacker_cfg = {
            "datasets": {
                "train": csv_sets["train_csv"],
                "dev": csv_sets["eval_csv"],
                "test": csv_sets["eval_csv"],
            },
            "classifier": classifier,
            "feature_extractors": [loss_features],
            "label": ["utt_in_set", "spk_in_set", "speaker_id"],
            "class_label": "spk_in_set",
            "batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers,
        }

        rank0_print("[MIA-Spk] Creating attacker...")
        attacker = SpeakerLevelAttacker(cfg=attacker_cfg)

        override = cfg.override

        rank0_print("[MIA-Spk] Training classifier...")
        attacker.train_classifier(override=override)

        eval_split = cfg.eval_split
        rank0_print(f"[MIA-Spk] Evaluating classifier on split='{eval_split}'...")
        results = attacker.evaluate(split=eval_split, override=override)

        rank0_print("[MIA-Spk] Results:")
        for k, v in results.items():
            rank0_print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")

        return results

    except Exception as e:
        rank0_print(f"[MIA-Spk] Error: {e}")
        traceback.print_exc()
        return {}

    finally:
        _cleanup_temp_dir(temp_dir, artifact_dir)
