from random import sample
from typing import List

import numpy as np
from scipy.spatial.distance import euclidean

Point = List[int]


def assign_cluster(point: Point, centroids: List[Point]) -> int:
    """
    return index of closest centroid for a given point
    """
    min_distance = float("inf")
    closest_centroid_idx = None
    for idx, centroid in enumerate(centroids):
        distance = euclidean(point, centroid)
        if distance < min_distance:
            min_distance = distance
            closest_centroid_idx = idx
    return closest_centroid_idx


def calculate_variation(points: List[Point]) -> float:
    """
    return the maximum distance between any two points in a list
    """
    max_distance = 0
    for i in range(len(points)):
        for j in range(len(points)):
            distance = euclidean(points[i], points[j])
            max_distance = distance if distance > max_distance else max_distance

    return max_distance


def within_threshold(
    centroids: List[Point], means: List[Point], threshold: float
) -> bool:
    for centroid, mean in zip(centroids, means):
        distance = euclidean(centroid, mean)
        if distance > threshold:
            return False

    return True


def kmeans(
    points: List[Point],
    k: int = 3,
    runs: int = 10,
    max_iter: int = 300,
    threshold: float = 0.0001,
) -> List[Point]:
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

    for i in range(runs):
        centroids = sample(points, k)
        for i in range(max_iter):
            clusters = [[] for c in centroids]
            for p in points:
                closest_centroid_idx = assign_cluster(p, centroids)
                clusters[closest_centroid_idx].append(p)

            means = [np.mean(c, axis=0).tolist() for c in clusters]
            if within_threshold(centroids, means, threshold):
                break

            centroids = means

        variation = sum([calculate_variation(c) for c in clusters])
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
