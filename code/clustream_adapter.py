
from dataclasses import dataclass
from datetime import datetime
import threading
import numpy as np
from collections import defaultdict


class MicroCluster:
    """
    CluStream-style micro-cluster with fading and thread-safe updates.
    """
    def __init__(self, cluster_id: int, point: np.ndarray, initial_user_id: str, decay_lambda: float = 0.001):
        self.cluster_id = cluster_id
        self.centroid = point.astype(float).copy()
        self.linear_sum = point.astype(float).copy()
        self.squared_sum = (point.astype(float) ** 2).copy()
        self.weight = 1.0
        self.n_points = 1
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self._user_ids = {initial_user_id}
        self._lock = threading.RLock()
        self.decay_lambda = decay_lambda
        self.variance = np.zeros_like(point, dtype=float)
        self.stability_score = 1.0

    def add_user_atomic(self, user_id: str) -> bool:
        with self._lock:
            was_new = user_id not in self._user_ids
            self._user_ids.add(user_id)
            return was_new

    def get_user_count(self) -> int:
        with self._lock:
            return len(self._user_ids)

    def get_user_ids_copy(self) -> set:
        with self._lock:
            return self._user_ids.copy()

    def get_true_size(self) -> int:
        return self.get_user_count()

    def _apply_fade(self, dt_seconds: float):
        if dt_seconds <= 0 or self.decay_lambda <= 0:
            return
        fade = np.exp(-self.decay_lambda * dt_seconds)
        self.weight *= fade
        self.linear_sum *= fade
        self.squared_sum *= fade

    def radius(self) -> float:
        if self.weight <= 0:
            return 0.0
        mean = self.linear_sum / self.weight
        var = np.maximum(self.squared_sum / self.weight - mean * mean, 0.0)
        return float(np.sqrt(np.sum(var)))

    def update(self, point: np.ndarray, user_id: str, learning_rate: float = 1.0):
        now = datetime.now()
        dt = (now - self.last_updated).total_seconds()
        with self._lock:
            self._apply_fade(dt)
            self.weight += learning_rate
            self.n_points += 1
            self.linear_sum += point
            self.squared_sum += point * point
            self.centroid = self.linear_sum / max(self.weight, 1e-9)
            mean_sq = self.squared_sum / max(self.weight, 1e-9)
            self.variance = np.maximum(mean_sq - self.centroid ** 2, 0.0)
            self.last_updated = now
            self.add_user_atomic(user_id)
            stab = 1.0 / (1.0 + float(np.mean(self.variance)))
            self.stability_score = 0.9 * self.stability_score + 0.1 * stab


class CluStreamOnline:
    """
    Lightweight CluStream implementation to replace FixedOnlineKMeans.
    """
    def __init__(self, max_clusters=50, min_cluster_size=15, init_radius=0.6, merge_threshold=0.4, decay_lambda=0.001):
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.init_radius = init_radius
        self.merge_threshold = merge_threshold
        self.decay_lambda = decay_lambda

        self.clusters: list[MicroCluster] = []
        self.n_features = None
        self._user_to_cluster: dict[str, int] = {}
        self._lock = threading.RLock()
        self.cluster_id_counter = 0
        self.total_points_seen = 0
        self.total_clustering_moves = 0

        self.user_drift_count = defaultdict(int)
        self.total_drift_events = 0
        self.drift_distances = []
        self.user_cluster_history = defaultdict(list)

    def _closest_cluster(self, point: np.ndarray):
        if not self.clusters:
            return None, float("inf")
        best, best_d = None, float("inf")
        for c in self.clusters:
            d = float(np.linalg.norm(point - c.centroid))
            if d < best_d:
                best, best_d = c, d
        return best, best_d

    def _merge_two_closest(self):
        if len(self.clusters) < 2:
            return False
        n = len(self.clusters)
        best_i, best_j, best_d = -1, -1, float("inf")
        for i in range(n):
            ci = self.clusters[i]
            for j in range(i + 1, n):
                cj = self.clusters[j]
                d = float(np.linalg.norm(ci.centroid - cj.centroid))
                if d < best_d:
                    best_d, best_i, best_j = d, i, j
        if best_i == -1:
            return False
        ci, cj = self.clusters[best_i], self.clusters[best_j]
        with ci._lock, cj._lock:
            w = ci.weight + cj.weight
            if w <= 0:
                new_centroid = (ci.centroid + cj.centroid) / 2.0
            else:
                new_centroid = (ci.centroid * ci.weight + cj.centroid * cj.weight) / w
            ci.linear_sum += cj.linear_sum
            ci.squared_sum += cj.squared_sum
            ci.weight += cj.weight
            ci.n_points += cj.n_points
            ci.centroid = new_centroid
            ci._user_ids.update(cj._user_ids)
            ci.last_updated = max(ci.last_updated, cj.last_updated)
        del self.clusters[best_j]
        return True

    def _record_drift_event(self, user_id: str, old_cluster_id: int, new_cluster_id: int, distance: float):
        self.user_drift_count[user_id] += 1
        self.total_drift_events += 1
        self.total_clustering_moves += 1
        self.drift_distances.append(distance)
        if len(self.drift_distances) > 1000:
            self.drift_distances = self.drift_distances[-500:]

    def fit_partial(self, point: np.ndarray, user_id: str) -> int:
        with self._lock:
            if self.n_features is None:
                self.n_features = len(point)
            self.total_points_seen += 1

            was_existing = user_id in self._user_to_cluster
            old_cluster_id = self._user_to_cluster.get(user_id)

            c, d = self._closest_cluster(point)
            assigned = None

            if c is not None and (d <= self.init_radius or c.radius() <= self.init_radius):
                assigned = c
            else:
                if len(self.clusters) >= self.max_clusters:
                    self._merge_two_closest()
                mc = MicroCluster(self.cluster_id_counter, point, user_id, decay_lambda=self.decay_lambda)
                self.cluster_id_counter += 1
                self.clusters.append(mc)
                assigned = mc

            assigned.update(point, user_id, learning_rate=1.0)

            if was_existing and old_cluster_id != assigned.cluster_id:
                self._record_drift_event(user_id, old_cluster_id, assigned.cluster_id, d)

            self._user_to_cluster[user_id] = assigned.cluster_id
            if not was_existing or old_cluster_id != assigned.cluster_id:
                self.user_cluster_history[user_id].append(assigned.cluster_id)

            return assigned.cluster_id

    def get_drift_stats(self) -> dict:
        """Return drift analytics in a shape compatible with callers expecting KMeans stats."""
        total_users = len(self._user_to_cluster)
        unique_drifting_users = len(self.user_drift_count)
        total_drift_events = int(self.total_drift_events)
        avg_drift_per_user = (total_drift_events / max(unique_drifting_users, 1)) if unique_drifting_users > 0 else 0.0
        max_drift_per_user = max(self.user_drift_count.values()) if self.user_drift_count else 0
        stable_users = total_users - unique_drifting_users
        avg_drift_distance = float(np.mean(self.drift_distances)) if self.drift_distances else 0.0
        drift_rate = total_drift_events / max(self.total_points_seen, 1)

        # Build a simple distance distribution (bins in euclidean distance)
        bins = [0.5, 1.0, 2.0]
        labels = ["0-0.5", "0.5-1.0", "1.0-2.0", ">=2.0"]
        counts = [0, 0, 0, 0]
        for d in self.drift_distances:
            if d < bins[0]:
                counts[0] += 1
            elif d < bins[1]:
                counts[1] += 1
            elif d < bins[2]:
                counts[2] += 1
            else:
                counts[3] += 1
        drift_distribution = dict(zip(labels, counts))

        # Identify "high drift" users as those with >= 3 moves (tweak if needed)
        high_drift_users = [u for u, c in self.user_drift_count.items() if c >= 3]

        # Also provide a ranked list of top drifters (user_id, count)
        top_drifters = sorted(self.user_drift_count.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'unique_drifting_users': unique_drifting_users,
            'total_drift_events': total_drift_events,
            'stable_users': stable_users,
            'avg_drift_per_user': avg_drift_per_user,
            'max_drift_per_user': max_drift_per_user,
            'high_drift_users': high_drift_users,
            'avg_drift_distance': avg_drift_distance,
            'drift_rate': drift_rate,
            'drift_distribution': drift_distribution,
            'top_drifters': top_drifters,
        }

    def get_clustering_stats(self) -> dict:
        with self._lock:
            total_unique_users = len(self._user_to_cluster)
            users_in_valid_clusters = 0
            users_in_small_clusters = 0
            valid_cluster_count = 0
            small_cluster_count = 0
            cluster_sizes = []

            for c in self.clusters:
                size = c.get_user_count()
                cluster_sizes.append(size)
                if size >= self.min_cluster_size:
                    valid_cluster_count += 1
                    users_in_valid_clusters += size
                else:
                    small_cluster_count += 1
                    users_in_small_clusters += size

            total_cluster_users = users_in_valid_clusters + users_in_small_clusters
            accounting_complete = (total_unique_users == total_cluster_users)

            return {
                'total_tracked_users': total_unique_users,
                'assigned_users': users_in_valid_clusters,
                'unassigned_users': users_in_small_clusters,
                'total_clusters': len(self.clusters),
                'valid_clusters': valid_cluster_count,
                'small_clusters': small_cluster_count,
                'cluster_sizes': cluster_sizes,
                'accounting_complete': accounting_complete,
                'total_cluster_users': total_cluster_users,
                'min_cluster_size_threshold': self.min_cluster_size
            }


class CluStreamFactory:
    @staticmethod
    def make(max_clusters=50, min_cluster_size=25, init_radius=0.6, merge_threshold=0.4, decay_lambda=0.001):
        return CluStreamOnline(
            max_clusters=max_clusters,
            min_cluster_size=min_cluster_size,
            init_radius=init_radius,
            merge_threshold=merge_threshold,
            decay_lambda=decay_lambda
        )


    # --- Compatibility helpers (optional) ---
    def get_move_counters(self) -> dict:
        """Return common move counters for compatibility with KMeans-based engines."""
        return {
            'total_clustering_moves': int(self.total_clustering_moves),
            'total_drift_events': int(self.total_drift_events),
            'unique_drifting_users': len(self.user_drift_count),
        }
