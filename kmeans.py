from typing import List

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

    # min_variation: float = infinity
    # best_cluster: List[Point] = None

    # for i in max_iter:
    #   randomly initialize k centroids of the same dimension as the points in List[Point]
    #   may want to add random_seed param for testing
    #   while true:
    #       iterate through the points and assign each to the nearest centroid
    #       determine the mean of each cluster and reassign the centroid value to the mean
    #       break when centroids converge (previous centroid values equal the new centroid values)

    #   calculate the sum of variation of the points in the current centroids' clusters
    #   if the sum of variation is less than min_variation, reassign min_variation and best_cluster

    # return best_cluster
