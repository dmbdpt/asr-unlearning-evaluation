from typing import List, Union
from collections import defaultdict

from tqdm import tqdm
from torch.utils.data import Subset


def downsample_dataset_by_duration(
    dataset,
    max_duration: float,
) -> List[int]:
    """Select indices such that cumulative duration per speaker does not exceed max_duration."""
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        if isinstance(base_dataset, Subset):
            inner_base, inner_indices = _get_nested_subset_info(dataset.dataset)
            candidate_indices = [inner_indices[i] for i in dataset.indices] if inner_indices else list(range(len(dataset)))
            base_dataset = inner_base
        else:
            candidate_indices = dataset.indices
    else:
        base_dataset = dataset
        candidate_indices = list(range(len(dataset)))

    speaker_duration = defaultdict(float)
    selected = []

    for local_idx in tqdm(candidate_indices, desc="Downsampling dataset by duration"):
        sample = base_dataset[local_idx]

        spk = sample.get("speaker_id", "")
        dur = float(sample.get("wav_lens", 0.0))

        if speaker_duration[spk] + dur <= max_duration:
            speaker_duration[spk] += dur
            selected.append(local_idx)

    return selected


def _get_nested_subset_info(dataset):
    """Helper to get the base dataset and flatten indices from nested Subsets."""
    if not isinstance(dataset, Subset):
        return dataset, None
    
    all_indices = []
    current = dataset
    
    while isinstance(current, Subset):
        if isinstance(current.indices, list):
            all_indices.append(current.indices)
        else:
            all_indices.append([i.item() for i in current.indices])
        current = current.dataset
    
    base_dataset = current
    
    if len(all_indices) == 1:
        combined_indices = all_indices[0]
    else:
        combined_indices = all_indices[0]
        for next_indices in all_indices[1:]:
            combined_indices = [combined_indices[i] for i in next_indices]
    
    return base_dataset, combined_indices

def get_avg_duration(dataset):
    """Calculate the average duration per speaker in a dataset."""
    test_spk_durations = defaultdict(float)
    for utt in tqdm(dataset, desc="Calculating average duration per speaker"):
        spk = utt.get("speaker_id", "")
        dur = utt.get("wav_lens", 0.0)
        test_spk_durations[spk] += dur
    
    if not test_spk_durations:
        return 0.0
    return sum(test_spk_durations.values()) / len(test_spk_durations)

def aggregate_results(metrics, losses, sets_to_evaluate):
    """Aggregate metrics and losses into a single results dictionary."""
    aggregated_results = {}
    for set_name in sets_to_evaluate:
        aggregated_results[set_name] = {}

        speakers = set()
        if metrics and set_name in metrics:
            speakers.update(metrics[set_name].keys())
        if losses and set_name in losses:
            speakers.update(losses[set_name].keys())
            
        for spk in speakers:
            aggregated_results[set_name][spk] = {}
            if metrics and set_name in metrics and spk in metrics[set_name]:
                aggregated_results[set_name][spk]["metrics"] = metrics[set_name][spk]
            if losses and set_name in losses and spk in losses[set_name]:
                aggregated_results[set_name][spk]["losses"] = losses[set_name][spk]
    return aggregated_results
