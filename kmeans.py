from random import sample
from typing import List

import numpy as np

Point = List[int]


def kmeans(points: List[Point], k: int = 3, max_iter: int = 100) -> List[Point]:
    """
    return a list of cluster centroids, given a list of points and the
    number of clusters to identify
    """
    if k > len(points):
        raise ValueError("number of clusters (k) must be less than number of points")

    dimensions = {len(p) for p in points}
    if len(dimensions) > 1:
        raise ValueError("points must all have the same dimensions")

    min_variation: float = float("inf")
    best_cluster: List[Point] = None

    for i in max_iter:
        centroids = sample(points, k)
        clusters = {c: [] for c in centroids}
        while True:
            for p in points:
                closest_centroid = assign_cluster(p, centroids)
                clusters[closest_centroid].append(p)

            means = [np.mean(c).tolist() for c in clusters.items()]
            if sorted(means) == sorted(centroids):
                # centroids converge when previous centroid values equal the new centroid values
                break

            centroids = means

        variation = sum([calculate_variation(c) for c in clusters.items()])
        if variation < min_variation:
            min_variation = variation
            best_cluster = centroids

    return best_cluster


if __name__ == "__main__":
    # test data are one-dimensional points with three clear centroids at 2, 12, and 22
    points = [[1], [2], [3], [11], [12], [13], [21], [22], [23]]
    centroids = kmeans(points, k=3)
    assert len(centroids) == 3
    assert [2] in centroids
    assert [12] in centroids
    assert [22] in centroids
