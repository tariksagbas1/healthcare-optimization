import numpy as np
from scipy.spatial import distance_matrix


def population_constrained_clustering(coords, populations, pop_limit):
    """
    coords: list of (x, y) tuples
    populations: list of population numbers
    pop_limit: maximum total population per cluster

    Returns: list of clusters
        Each cluster is a list of node indices
    """
    coords = np.array(coords)
    populations = np.array(populations)
    n = len(coords)

    unassigned = set(range(n))
    clusters = []

    # Precompute the full distance matrix
    dist_matrix = distance_matrix(coords, coords)

    while unassigned:
        # Start a new cluster
        cluster = []
        cluster_pop = 0

        # Pick a random unassigned node to start
        current = unassigned.pop()
        cluster.append(current)
        cluster_pop += populations[current]

        while True:
            if not unassigned:
                break

            # Find the nearest unassigned neighbor
            nearest_neighbor = min(
                unassigned,
                key=lambda x: dist_matrix[current, x]
            )

            # If adding the neighbor exceeds population limit, stop this cluster
            if cluster_pop + populations[nearest_neighbor] > pop_limit:
                break

            # Otherwise, add the neighbor
            unassigned.remove(nearest_neighbor)
            cluster.append(nearest_neighbor)
            cluster_pop += populations[nearest_neighbor]

            # Move current node
            current = nearest_neighbor

        clusters.append(cluster)

    return clusters
