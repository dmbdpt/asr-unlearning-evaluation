import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, Sequence, List, Union

import numpy as np

import torch
import torch.distributed as dist

from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def compute_semantic_distance(
    target_features: np.ndarray,
    candidate_features: np.ndarray,
    distance_metric: str = 'cosine'
) -> Tuple[float, Dict[str, Any]]:

    if target_features is None or candidate_features is None:
        return 0.0, {'error': 'None input features'}

    if isinstance(target_features, torch.Tensor):
        target_features = target_features.detach().cpu().numpy()
    if isinstance(candidate_features, torch.Tensor):
        candidate_features = candidate_features.detach().cpu().numpy()

    if target_features.ndim == 1:
        target_features = target_features.reshape(1, -1)

    if candidate_features.ndim == 2:
        if candidate_features.shape[0] < candidate_features.shape[1]:
            candidate_features = candidate_features.reshape(candidate_features.shape[0], 1, -1)
        else:
            candidate_features = candidate_features.reshape(1, 1, -1)

    target_emb_avg = np.mean(target_features, axis=0, keepdims=True)
    target_norm = np.linalg.norm(target_emb_avg)

    if target_norm == 0:
        return 0.0, {'error': 'Zero target vector'}

    all_distances = []

    if distance_metric == 'cosine':
        for candidate_emb in candidate_features:
            candidate_emb_avg = np.mean(candidate_emb, axis=0, keepdims=True)
            if np.linalg.norm(candidate_emb_avg) == 0:
                all_distances.append(0.0)
            else:
                dists = cosine_distances(target_emb_avg, candidate_emb_avg)
                all_distances.append(float(dists[0, 0]))

    elif distance_metric == 'euclidean':
        for candidate_emb in candidate_features:
            candidate_emb_avg = np.mean(candidate_emb, axis=0, keepdims=True)
            if np.linalg.norm(candidate_emb_avg) == 0:
                all_distances.append(0.0)
            else:
                dists = euclidean_distances(target_emb_avg, candidate_emb_avg)
                all_distances.append(float(dists[0, 0]))

    else:
        return 0.0, {'error': f'Unknown distance metric: {distance_metric}'}

    all_distances = np.array(all_distances) if all_distances else np.array([0.0])
    mean_dist = float(np.mean(all_distances))
    min_dist = float(np.min(all_distances))
    max_dist = float(np.max(all_distances))
    std_dist = float(np.std(all_distances))

    detailed_distances = {
        'mean_distance': mean_dist,
        'min_distance': min_dist,
        'max_distance': max_dist,
        'std_distance': std_dist,
        'all_distances': all_distances.tolist(),
        'num_candidates': len(all_distances)
    }

    return mean_dist, detailed_distances


def compute_all_speaker_distances(
    data_handler,
    forget_speaker: str,
    dataset_name: str = "train"
) -> Dict[str, float]:
    """Compute semantic distances from forget speaker to all other speakers."""
    if dataset_name not in data_handler.datasets_features:
        return {}

    features = data_handler.datasets_features[dataset_name]
    speaker_map = data_handler.speaker_map_per_set.get(dataset_name, {})

    # Speaker IDs may be stored as either strings or integers depending on the dataset
    target_idx = None
    if forget_speaker in speaker_map:
        target_idx = speaker_map[forget_speaker]
    elif str(forget_speaker) in speaker_map:
        target_idx = speaker_map[str(forget_speaker)]
    elif forget_speaker.isdigit() and int(forget_speaker) in speaker_map:
        target_idx = speaker_map[int(forget_speaker)]

    if target_idx is None:
        return {}

    target_features = features[target_idx]

    if target_features.ndim > 1:
        target_features_avg = np.mean(target_features, axis=0, keepdims=True)
    else:
        target_features_avg = target_features.reshape(1, -1)

    distances = {}
    for speaker, idx in speaker_map.items():
        speaker_key = str(speaker)
        if speaker_key == str(forget_speaker):
            distances[speaker_key] = 0.0
            continue

        candidate_features = features[idx]
        if candidate_features.ndim > 1:
            candidate_features = np.mean(candidate_features, axis=0, keepdims=True)
        else:
            candidate_features = candidate_features.reshape(1, -1)

        distance, _ = compute_semantic_distance(target_features_avg, candidate_features)
        distances[speaker_key] = distance

    return distances


@dataclass
class ClusterMembershipResult:
    target_subject: str
    cluster_members: Sequence[Optional[str]]
    forget_set_original: Sequence[Optional[str]]
    retain_set: Sequence[Optional[str]]
    distances: Dict[str, float]
    cluster_size: int
    is_single: bool
    has_cluster: bool
    error: Optional[str] = None

    @property
    def forget_set(self) -> List[str]:
        if self.is_single or not self.has_cluster:
            return [self.target_subject]
        return self.forget_set_original

def compute_cluster_membership(
    data_handler,
    target_subject: str,
    clustering_config: dict,
    dataset_name: str = "train"
) -> ClusterMembershipResult:
    logger = logging.getLogger(__name__)

    distance_config = clustering_config.distance_config
    threshold_type = distance_config.threshold_type
    threshold_value = distance_config.threshold_value

    distances = compute_all_speaker_distances(
        data_handler=data_handler,
        forget_speaker=target_subject,
        dataset_name=dataset_name
    )

    if not distances:
        speaker_map = data_handler.speaker_map_per_set.get(dataset_name, {})
        logger.warning("[Cluster] No distances computed for target %s", target_subject)
        logger.warning("[Cluster] Debug - dataset_name in datasets_features: %s",
                       dataset_name in data_handler.datasets_features)
        logger.warning("[Cluster] Debug - target in speaker_map: %s", target_subject in speaker_map)
        logger.warning("[Cluster] Debug - speaker_map keys: %s",
                       list(speaker_map.keys())[:10] if speaker_map else "empty")
        return ClusterMembershipResult(
            target_subject=target_subject,
            cluster_members=[],
            forget_set_original=[],
            retain_set=[],
            distances={},
            cluster_size=0,
            is_single=False,
            has_cluster=False,
            error='No distances computed'
        )

    speaker_map = data_handler.speaker_map_per_set.get(dataset_name, {})
    all_speakers = list(speaker_map.keys())

    logger.info("[Cluster] Computing cluster membership for target: %s", target_subject)
    logger.info("[Cluster] Threshold type: %s, threshold value: %s", threshold_type, threshold_value)
    logger.info("[Cluster] Total speakers in dataset: %s", len(all_speakers))

    cluster_members = []

    if threshold_type == 'quantity':
        # threshold_value < 1 is treated as a fraction of all speakers, not a count
        if threshold_value < 1:
            num_to_select = int(threshold_value * len(all_speakers))
        else:
            num_to_select = int(threshold_value)

        # target always has distance 0, so it naturally sorts first
        sorted_speakers = sorted(distances.items(), key=lambda item: item[1])

        cluster_members = [target_subject]
        for speaker, distance in sorted_speakers:
            if speaker != target_subject and len(cluster_members) < num_to_select:
                cluster_members.append(speaker)

        logger.info("[Cluster] Selected %s speakers based on quantity threshold", len(cluster_members))

    elif threshold_type == 'distance':
        cluster_members = [target_subject]
        for speaker, distance in distances.items():
            if speaker != target_subject and distance < threshold_value:
                cluster_members.append(speaker)

        logger.info(
            "[Cluster] Selected %s speakers based on distance threshold < %s", len(cluster_members), threshold_value)
    elif threshold_type == 'sequential':
        if threshold_value < 1:
            num_to_select = int(threshold_value * len(all_speakers))
        else:
            num_to_select = int(threshold_value)
        # Select speakers sequentially until distance exceeds threshold
        cluster_members = [target_subject]  # Always include target
        seq_speakers = data_handler.speaker_map_per_set["train"].keys()
        for speaker in seq_speakers:
            if speaker != target_subject and len(cluster_members) < num_to_select:
                cluster_members.append(speaker)

        logger.info("[Cluster] Selected %s speakers based on sequential threshold",
                    len(cluster_members))
        logger.info("[Cluster] Sequential speakers: %s", cluster_members)
    else:
        logger.error("[Cluster] Unknown threshold type: %s", threshold_type)
        return ClusterMembershipResult(
            target_subject=target_subject,
            cluster_members=[],
            forget_set_original=[],
            retain_set=[],
            distances=distances,
            cluster_size=0,
            is_single=False,
            has_cluster=False,
            error=f'Unknown threshold type: {threshold_type}'
        )

    retain_set = [s for s in all_speakers if s not in cluster_members]

    is_single = len(cluster_members) == 1
    # A single-subject cluster is still valid for unlearning
    has_cluster = len(cluster_members) >= 1

    logger.info("[Cluster] Target subject: %s", target_subject)
    logger.info("[Cluster] Cluster members (%s): %s", len(cluster_members), cluster_members)
    logger.info("[Cluster] Retain set (%s): %s", len(retain_set), retain_set)

    if is_single:
        logger.warning("[Cluster] WARNING: Cluster contains only the target subject (%s)", target_subject)

    if not has_cluster:
        logger.warning("[Cluster] WARNING: No cluster found for target %s", target_subject)

    for member in cluster_members:
        distance = distances.get(member, None)
        if distance is not None:
            logger.info("[Cluster] Member %s: distance = %.4f", member, distance)

    return ClusterMembershipResult(
        target_subject=target_subject,
        cluster_members=cluster_members,
        forget_set_original=cluster_members,  # Same as cluster_members
        retain_set=retain_set,
        distances=distances,
        cluster_size=len(cluster_members),
        is_single=is_single,
        has_cluster=has_cluster
    )


def rank0_print(*args, **kwargs):
    """Print only if the current process is rank 0."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def compute_deltas(
    pre_metrics: Dict[str, Dict[str, Union[int, float]]],
    post_metrics: Dict[str, Dict[str, Union[int, float]]]
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Compute delta metrics between pre and post evaluation results."""
    speaker_deltas = []
    avg_deltas = {}

    # Per-speaker deltas
    common_speakers = set(pre_metrics.keys()) & set(post_metrics.keys())
    for speaker_id in common_speakers:
        pre_speaker = pre_metrics[speaker_id]
        post_speaker = post_metrics[speaker_id]
        common_metrics = set(pre_speaker.keys()) & set(post_speaker.keys())

        for metric_name in common_metrics:
            delta = post_speaker[metric_name] - pre_speaker[metric_name]
            speaker_deltas.append({
                "speaker_id": speaker_id,
                "metric_name": metric_name,
                "delta": delta
            })

    # Average deltas across all speakers
    if pre_metrics:
        all_metric_names = set()
        for m in pre_metrics.values():
            all_metric_names.update(m.keys())

        for metric_name in all_metric_names:
            pre_values = [m[metric_name] for m in pre_metrics.values() if metric_name in m]
            post_values = [m[metric_name] for m in post_metrics.values() if metric_name in m]

            if pre_values and post_values:
                pre_avg = sum(pre_values) / len(pre_values)
                post_avg = sum(post_values) / len(post_values)
                avg_deltas[metric_name] = post_avg - pre_avg

    return speaker_deltas, avg_deltas


def extract_speaker_metrics(
    set_data: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Union[int, float]]]:
    """Extract numeric metrics from evaluation results organized by speaker."""
    metrics_by_speaker: Dict[str, Dict[str, Union[int, float]]] = {}

    for speaker_id, speaker_data in set_data.items():
        if not isinstance(speaker_data, dict):
            continue

        speaker_metrics: Dict[str, Union[int, float]] = {}
        for metric_name, metric_value in speaker_data.items():
            if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                speaker_metrics[metric_name] = metric_value

        if speaker_metrics:
            metrics_by_speaker[str(speaker_id)] = speaker_metrics

    return metrics_by_speaker
