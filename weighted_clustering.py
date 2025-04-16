from math import dist, sqrt
from sklearn.cluster import AgglomerativeClustering
from time import time

def d_cluster(dataset_index):
    # Initial data (unchanged)
    p = []  # Population of every community
    cord_com = []  # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file:  # Read from file
        lines = file.readlines()

    line1 = lines[0].split()
    n = int(line1[0])  # Number of nodes
    m = int(line1[1])  # Number of healthcare units to place

    for line in lines[2:]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, C, population = values[1], values[2], values[3], int(values[4])
        cord_com.append([x, y])  # add coordinates
        p.append(population)  # add populations for each community i

    # Calculate distances
    d_ij = []
    for x1, y1 in cord_com:
        row = []
        for x2, y2 in cord_com:
            distance = dist((x1, y1), (x2, y2))
            row.append(distance)

        d_ij.append(row)

    # create matrix of weighted distances
    w_d_ij = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(d_ij[i][j]*(p[i]+p[j]) / 2)

        w_d_ij.append(row)

    start = time()

    # Step 1: Fit clustering on distance matrix
    import numpy as np
    w_d_ij = np.array(w_d_ij)
    clustering = AgglomerativeClustering(n_clusters=m, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(w_d_ij)

    # Step 2: Group node indices by cluster
    clusters = [[] for _ in range(m)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    # Step 3: For each cluster, find the most "central" node
    # (the one with minimum total distance to all others in its cluster)
    closest_nodes = []

    for group in clusters:
        submatrix = w_d_ij[np.ix_(group, group)]  # cluster's intra-distance matrix
        total_dist = submatrix.sum(axis=1)  # total distance from each node to others
        best_idx_in_cluster = group[np.argmin(total_dist)]
        closest_nodes.append(best_idx_in_cluster)
    end = time()
    return closest_nodes, round(end - start, 2)
