import numpy as np
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics.pairwise import cosine_distances

from src.data.features.feature_extractor import FeatureExtractor
from src.data.data_handler import DataHandler


class Clusterer:
    def __init__(self, feature_extractor: FeatureExtractor = None, data_handler: DataHandler = None, cfg: dict = None):
        self.feature_extractor = feature_extractor
        self.data_handler = data_handler
        self.cfg = cfg

        self.distances = None

        self.distance_calc = cfg['distance_calc']
        self.sum_normalized = cfg['sum_normalized']

        self.feature_dims = self.feature_extractor.get_feature_dims()

    def cluster(self, features: dict = None, dataset_name: str = "train", target: str = None):

        if self.cfg['type'] == 'distance':
            return self._cluster_by_distance(dataset_name=dataset_name, target=target)
        elif self.cfg['type'] == 'clustering':
            raise NotImplementedError("Clustering-based method is not implemented yet.")
        else:
            raise ValueError(f"Invalid clustering type: {self.cfg['type']}")

    def _cluster_by_distance(self, dataset_name: str = "train", target: str = None):
        cfg_distance = self.cfg['distance_config']

        threshold_type = cfg_distance['threshold_type']
        threshold = cfg_distance['threshold_value']

        speaker_distances = {}

        target_features = self.data_handler.get_subject_features(target, dataset=dataset_name)

        candidate_speakers = [s for s in self.data_handler.data["all_speakers"] if s != target]

        for speaker in candidate_speakers:
            try:
                # Use global lookup (dataset=None) to find candidate in any dataset
                candidate_features = self.data_handler.get_subject_features(speaker, dataset=None)
            except (KeyError, ValueError):
                continue

            # Pairwise distances: (N_target, N_candidate)
            t_feats = self.feature_extractor.separate_features(target_features)
            c_feats = self.feature_extractor.separate_features(candidate_features)

            if isinstance(t_feats, tuple) and isinstance(c_feats, tuple):
                d_acoustic = cosine_distances(t_feats[0], c_feats[0])
                d_textual = cosine_distances(t_feats[1], c_feats[1])

                if np.max(d_acoustic) > 0:
                    d_acoustic = d_acoustic / np.max(d_acoustic)
                if np.max(d_textual) > 0:
                    d_textual = d_textual / np.max(d_textual)

                dists = d_acoustic + d_textual
            else:
                dists = cosine_distances(target_features, candidate_features)

            # single linkage: distance between two speakers is their closest pair of samples
            min_dist = np.min(dists)

            speaker_distances[speaker] = min_dist

        self.distances = speaker_distances
        self.data_handler.save_distances(speaker_distances, target, dataset_name=dataset_name)

        if threshold_type == 'quantity':
            if threshold < 1:
                threshold = int(threshold * len(speaker_distances))

            sorted_speakers = sorted(speaker_distances.items(), key=lambda item: item[1])
            selected_speakers = [s[0] for s in sorted_speakers[:threshold]]
            return selected_speakers
        elif threshold_type == 'distance':
            selected_speakers = [s for s, d in speaker_distances.items() if d < threshold]
            return selected_speakers

        return []
