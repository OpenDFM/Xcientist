from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np


class ClusterType(Enum):

    POTENTIAL = "PMC"
    OUTLIER = "OMC"


@dataclass
class MicroCluster:
    """DenStream micro-cluster representation."""

    id: int
    kind: ClusterType
    linear_sum: np.ndarray
    square_sum: np.ndarray
    weight: float
    last_update: float
    v_sum: np.ndarray           
    W_sum: float
    S_sum: float
    r_ema: float

    def decay(self, factor: float) -> None:
        """Apply exponential decay to the cluster statistics."""
        if factor == 1.0:
            return
        self.weight *= factor
        self.linear_sum *= factor
        self.square_sum *= factor

    def update(self, point: np.ndarray) -> None:
        """Absorb a new point into the cluster."""
        self.linear_sum += point
        self.square_sum += point * point
        self.weight += 1.0

        u = point / (np.linalg.norm(point) + 1e-12)
        self.v_sum += u
        self.W_sum += 1.0
        self.S_sum += 1.0

        center_u = self.v_sum / (np.linalg.norm(self.v_sum) + 1e-12)
        dev = max(0.0, 1.0 - float(np.dot(center_u, u)))
        self.r_ema = 0.9 * self.r_ema + 0.1 * dev
    
    def coherence_R(self) -> float:
        return float(np.linalg.norm(self.v_sum) / (self.W_sum + 1e-12))
 
    def avg_pairwise_cos(self) -> float:
        num = float(self.v_sum @ self.v_sum) - self.S_sum
        den = float(self.W_sum * self.W_sum - self.S_sum)
        return 1.0 if den <= 1e-12 else max(-1.0, min(1.0, num / den))

    @property
    def center(self) -> np.ndarray:
        """Current cluster center."""
        if self.weight <= 0.0:
            raise ValueError("Cluster weight must be positive to compute center.")
        return self.linear_sum / self.weight

    def radius(self) -> float:
        """Compute the root mean square deviation as cluster radius."""
        if self.weight <= 1.0:
            return 0.0
        ls_mean = self.linear_sum / self.weight
        ss_mean = self.square_sum / self.weight
        variance = np.maximum(ss_mean - ls_mean ** 2, 0.0)
        return float(np.sqrt(np.sum(variance)))


class DenStream:
    def __init__(self, eps: float = 0.6, beta: float = 0.5, mu: float = 4) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must be in (0, 1).")
        if mu <= 0:
            raise ValueError("mu must be positive.")

        self.eps = float(eps)
        self.beta = float(beta)
        self.mu = float(mu)

        self.potential_clusters: List[MicroCluster] = []
        self.outlier_clusters: List[MicroCluster] = []

        self.cidmap2cluster: Dict[int, MicroCluster] = {}

        self._cluster_id = 0

    def process(self, point: np.ndarray, now: Optional[str] = None) -> Dict[str, object]:
        """
        Absorb a new data point from the stream.

        Returns a dictionary summarizing the update, including promotion events and
        cleanup removals if any occurred.
        """
        summary = {"time": now}
        if point.ndim != 1:
            raise ValueError("point must represent a 1D vector.")

        target = self._nearest_cluster(point, self.potential_clusters, now)
        if target and self._absorb(target, point, now):
            summary["absorbed_into"] = {"type": target.kind.value, "cluster_id": target.id}
            summary["cluster_weight"] = float(target.weight)
        else:
            target = self._nearest_cluster(point, self.outlier_clusters, now)
            if target and self._absorb(target, point, now):
                summary["absorbed_into"] = {
                    "type": target.kind.value,
                    "cluster_id": target.id,
                }
                summary["cluster_weight"] = float(target.weight)
                if target.weight >= self.beta * self.mu and target.kind is ClusterType.OUTLIER:
                    self._promote(target)
                    summary["promoted"] = [target.id]
                    summary["absorbed_into"]["type"] = ClusterType.POTENTIAL.value
            else:
                target = self._create_outlier(point, now)
                summary["absorbed_into"] = {"type": "NEW_OMC", "cluster_id": target.id}
                summary["cluster_weight"] = float(target.weight)

        return summary

    def _nearest_cluster(self, point: np.ndarray, clusters: List[MicroCluster], now: float) -> Optional[MicroCluster]:
        candidate: Optional[MicroCluster] = None
        best_distance = self.eps
        for cluster in list(clusters):
            if cluster.weight < 1e-9:
                continue
            distance = float(np.linalg.norm(point - cluster.center))
            if distance <= self.eps and distance < best_distance:
                candidate = cluster
                best_distance = distance
        return candidate

    def _absorb(self, cluster: MicroCluster, point: np.ndarray, now: float) -> bool:
        previous_state = (cluster.weight, cluster.linear_sum.copy(), cluster.square_sum.copy())
        cluster.update(point)
        cluster.last_update = now

        if cluster.kind is ClusterType.POTENTIAL and cluster.radius() > self.eps:
            cluster.weight, cluster.linear_sum, cluster.square_sum = previous_state
            cluster.last_update = now
            return False
        return True
    
    def _unit(self, x: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(x))
        return x if n == 0.0 else (x / (n + 1e-12))

    def _create_outlier(self, point: np.ndarray, now: float) -> MicroCluster:
        u = self._unit(point)
        cluster = MicroCluster(
            id=self._next_cluster_id(),
            kind=ClusterType.OUTLIER,
            linear_sum=point.copy(),
            square_sum=point * point,
            weight=1.0,
            last_update=now,
            v_sum=u.copy(),
            W_sum=1.0, 
            S_sum=1.0,  
            r_ema=0.0, 
        )
        self.outlier_clusters.append(cluster)
        self.cidmap2cluster[cluster.id] = cluster
        return cluster

    def _promote(self, cluster: MicroCluster) -> None:
        cluster.kind = ClusterType.POTENTIAL
        self.outlier_clusters = [c for c in self.outlier_clusters if c.id != cluster.id]
        self.potential_clusters.append(cluster)

    def _cleanup(self, now: float) -> Dict[str, List[int]]:
        removed = {"omc": [], "pmc": []}
        omc_threshold = self.beta * self.mu
        updated_outliers: List[MicroCluster] = []
        for cluster in self.outlier_clusters:
            if cluster.weight < omc_threshold:
                removed["omc"].append(cluster.id)
            else:
                updated_outliers.append(cluster)
        self.outlier_clusters = updated_outliers

        pmc_threshold = self.mu
        updated_potential: List[MicroCluster] = []
        for cluster in self.potential_clusters:
            if cluster.weight < pmc_threshold:
                removed["pmc"].append(cluster.id)
            else:
                updated_potential.append(cluster)
        self.potential_clusters = updated_potential
        return removed

    def get_micro_clusters(self) -> Dict[str, List[Tuple[int, np.ndarray, float]]]:
        """Return current micro-clusters for introspection."""
        pmc = [(c.id, c.center.copy(), float(c.weight)) for c in self.potential_clusters]
        omc = [(c.id, c.center.copy(), float(c.weight)) for c in self.outlier_clusters]
        return {"pmc": pmc, "omc": omc}

    def _next_cluster_id(self) -> int:
        self._cluster_id += 1
        return self._cluster_id


def _demo() -> None:
    """Minimal runnable example demonstrating OMC promotion to PMC."""
    denstream = DenStream(eps=0.6, beta=0.5, mu=4)
    stream = [
        (0.0, 0.0),
        (0.1, 0.1),
        (-0.05, 0.0),
        (0.05, -0.05),
        (2.5, 2.5),
        (2.4, 2.6),
    ]

    print("Streaming points through DenStream:")
    for point in stream:
        info = denstream.process(point, now='test')
        cluster_info = info["absorbed_into"]
        cluster_id = cluster_info["cluster_id"]
        cluster_type = cluster_info["type"]
        weight = info.get("cluster_weight", 0.0)
        if "promoted" in info:
            print(f"    promotion: OMC #{info['promoted'][0]} became a PMC")
        if "cleanup" in info:
            removed = info["cleanup"]
            if removed["pmc"] or removed["omc"]:
                print(f"    cleanup removed pmc={removed['pmc']} omc={removed['omc']}")

    print("\nFinal micro-clusters:")
    clusters = denstream.get_micro_clusters()
    for kind, items in clusters.items():
        for cluster_id, center, weight in items:
            center_tuple = tuple(round(float(x), 3) for x in center)
            print(f"  {kind} #{cluster_id:02d} center={center_tuple} weight={weight:.2f}")


if __name__ == "__main__":
    _demo()
