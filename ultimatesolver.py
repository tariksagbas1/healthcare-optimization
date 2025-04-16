from gurobipy import Model, GRB, quicksum
from math import dist, sqrt
from numpy import percentile, linspace, array
from time import time
from matplotlib import pyplot
from sklearn.cluster import KMeans, AgglomerativeClustering
from maybe_lp_relax import lp_global

"""
CLUSTERING
"""
def cluster(dataset_index):
    p = []  # Population of every community
    cord_com = []  # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file:  # Read from file
        lines = file.readlines()

    line1 = lines[0].split()
    n = int(line1[0])  # Number of nodes
    m = int(line1[1])  # Number of healthcare units to place
    M = n

    for line in lines[2:]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, C, population = values[1], values[2], values[3], int(values[4])
        cord_com.append([x, y])  # add coordinates
        p.append(population)  # add populations for each community i
    start = time()
    kmeans = KMeans(n_clusters=m, random_state=0, n_init=20).fit(cord_com)
    facility_coords = kmeans.cluster_centers_
    assigned_centers = kmeans.predict(cord_com)
    from scipy.spatial.distance import cdist
    import numpy as np

    facility_indices = []
    for center in facility_coords:
        dists = cdist([center], cord_com)
        closest_index = int(np.argmin(dists))
        facility_indices.append(closest_index)
    end = time()
    cluster_time = round(end - start, 2)
    return [facility_indices , cluster_time]

def d_cluster(dataset_index):
    p = []  # Population of every community
    cord_com = []  # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file:  # Read from file
        lines = file.readlines()

    line1 = lines[0].split()
    n = int(line1[0])  # Number of
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
    labels = clustering.fit_predict(w_d_ij)  # D is your (n x n) distance matrix

    # Step 2: Group node indices by cluster
    clusters = [[] for _ in range(m)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    # Step 3: For each cluster, find the most "central" node
    # (i.e., the one with minimal total distance to all others in its cluster)
    closest_nodes = []

    for group in clusters:
        submatrix = w_d_ij[np.ix_(group, group)]  # cluster's intra-distance matrix
        total_dist = submatrix.sum(axis=1)  # total distance from each node to others
        best_idx_in_cluster = group[np.argmin(total_dist)]
        closest_nodes.append(best_idx_in_cluster)
    end = time()
    return closest_nodes, round(end - start, 2)


def sam_ifs2(dataset_index):
    # INITIAL DATA

    p = []  # Population of every community
    cord_com = []  # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file:  # Read from file
        lines = file.readlines()

    line1 = lines[0].split()
    n = int(line1[0])  # Number of
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

    units, cluster_time = d_cluster(dataset_index)

    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    b_ij = {}  # b_ij = 1 if community i is served by unit on j
    for unit_index in units:
        for i in range(n):
            b_ij[(i, unit_index)] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{unit_index}")

    # Z: auxiliary variable
    Z = model.addVar(vtype=GRB.CONTINUOUS, name="Z")

    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    # Z constraint
    for unit_index in units:
        for i in range(n):
            if i != unit_index:
                model.addConstr(Z >= b_ij[(i, unit_index)] * d_ij[i][unit_index] * p[i],
                                name=f"Z_constraint_{i}_{unit_index}")

    # Single allocation constraint, sum(b_ij) == 1 for every population
    for i in range(n):
        variables = 0.0
        for unit_index in units:
            variables += b_ij[(i, unit_index)]
        model.addConstr(variables == 1, name=f"single_allocation_constraint_{i}")

    # Capacity constraint
    for unit_index in units:
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, unit_index)] * p[i]
        model.addConstr(variables <= C, name=f"capacity_constraint_{unit_index}")

    model.update()
    model.setParam(GRB.Param.MIPGap, 0.05)

    # START TIMER
    start = time()

    # Solve Model
    model.optimize()

    print()
    # END TIMER
    end = time()
    time_elapsed = end - start
    if model.status == GRB.OPTIMAL:
        z_val = Z.x
        for unit_index in units:
            for i in range(n):
                b_ij[(i, unit_index)] = b_ij[(i, unit_index)].x

        return b_ij, cluster_time + round(time_elapsed, 2), z_val
    else:
        return "Model Is Infeasible"

def sam_global(dataset_index):
    # INITIAL DATA

    p = []  # Population of every community
    cord_com = []  # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file:  # Read from file
        lines = file.readlines()

    line1 = lines[0].split()
    n = int(line1[0])  # Number of
    m = int(line1[1])  # Number of healthcare units to place
    M = n

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

    minimums = []
    for i in range(n):
        weight_dist = set()
        for j in range(n):
            if i != j:
                weight_dist.add(p[i] * d_ij[i][j])
        minimums.append(min(weight_dist))

    maximums = []
    for i in range(n):
        index = -1
        max_val = -1
        for j in range(n):
            if i != j:
                val = p[i] * d_ij[i][j]
                if val > max_val:
                    max_val = val
                    index = j
        maximums.append((i, index, max_val))
    maximums.sort(key=lambda x: x[2], reverse=True)
    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: 1 if population i is served by unit on j 0 o/w
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}")

    # z_j: 1 if facility is opened at node j. 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype=GRB.BINARY, name=f"z_{j}"))

    # Z: auxiliary variable
    Z = model.addVar(vtype=GRB.CONTINUOUS, name="Z")

    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints
    for j in range(n):
        model.addConstr(Z >= minimums[j] * (1 - z_j[j]))

    model.addConstr(Z <= maximums[0][2])

    # Z > b_ij * d_ij for every i, j
    for i in range(n):
        for j in range(n):
            model.addConstr(Z >= p[i] * b_ij[i, j] * d_ij[i][j], name=f"Z_constraint_{i}_{j}")

    # Capacity constraint, sum(b_ij * p[i]) <= C for every unit. (There can be surplus capacity ?)
    for j in range(n):
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, j)] * p[i]
        model.addConstr(variables <= C * z_j[j], name=f"capacity_constraint_{i}")

    # Population constraint
    for i in range(n):
        sum = 0.0
        for j in range(n):
            sum += b_ij[(i, j)]
        model.addConstr(1 == sum, name=f"Population_constraint{i}")

    # Total unit constraint, sum(x_i) == m

    g_j = []
    for j in range(n):
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, j)]
        g_j.append(variables)
        model.addConstr(g_j[j] <= M * z_j[j])

    sum = 0.0
    for j in range(n):
        sum += z_j[j]
    model.addConstr(sum == m, name=f"unit_constraint")

    best_Z = None
    # START TIMER
    start = time()

    # Solve Model
    model.optimize()

    print()
    # END TIMER
    end = time()
    time_elapsed = end - start
    if model.status == GRB.OPTIMAL:
        assignments = {}
        for j in range(n):
            coms = []
            for i in range(n):
                if b_ij[(i, j)]:  ################## MODIFIED ####################
                    if b_ij[(i, j)].x != 0:
                        coms.append(i)
            if coms != []:
                assignments[j] = coms

        ls = []
        for i in range(n):
            for j in range(n):
                if b_ij[(i, j)].x == 1 and j not in ls:
                    ls.append(j)
        return str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x), ls, assignments
    else:
        return "Model Is Infeasible"

def sam_global_w_ifs2(dataset_index):
    # INITIAL DATA

    p = []  # Population of every community
    cord_com = []  # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file:  # Read from file
        lines = file.readlines()

    line1 = lines[0].split()
    n = int(line1[0])  # Number of
    m = int(line1[1])  # Number of healthcare units to place
    M = n

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

    minimums = []
    for i in range(n):
        weight_dist = set()
        for j in range(n):
            if i != j:
                weight_dist.add(p[i] * d_ij[i][j])
        minimums.append(min(weight_dist))

    maximums = []
    for i in range(n):
        index = -1
        max_val = -1
        for j in range(n):
            if i != j:
                val = p[i] * d_ij[i][j]
                if val > max_val:
                    max_val = val
                    index = j
        maximums.append((i, index, max_val))
    maximums.sort(key=lambda x: x[2], reverse=True)

    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: if population i is served by unit on j
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}")

    # z_j: 1 if facility is opened at node j. 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype=GRB.BINARY, name=f"z_{j}"))

    # Z: auxiliary variable
    Z = model.addVar(vtype=GRB.CONTINUOUS, name="Z")

    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    for j in range(n):
        model.addConstr(Z >= minimums[j] * (1 - z_j[j]))

    model.addConstr(Z <= maximums[0][2])

    # Z > b_ij * d_ij for every i, j
    for i in range(n):
        for j in range(n):
            model.addConstr(Z >= p[i] * b_ij[i, j] * d_ij[i][j], name=f"Z_constraint_{i}_{j}")

    # Capacity constraint, sum(b_ij * p[i]) <= C for every unit. (There can be surplus capacity ?)
    for j in range(n):
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, j)] * p[i]
        model.addConstr(variables <= C * z_j[j], name=f"capacity_constraint_{i}")

    # Population constraint
    for i in range(n):
        sum = 0.0
        for j in range(n):
            sum += b_ij[(i, j)]
        model.addConstr(1 == sum, name=f"Population_constraint{i}")

    # Total unit constraint, sum(x_i) == m
    g_j = []
    for j in range(n):
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, j)]
        g_j.append(variables)
        model.addConstr(g_j[j] <= M * z_j[j])

    sum = 0.0
    for j in range(n):
        sum += z_j[j]
    model.addConstr(sum == m, name=f"unit_constraint")

    # START TIMER
    start = time()


    b_ij_start, ifs_time, z_val = sam_ifs2(dataset_index)

    model.addConstr(Z <= z_val)
    # Set starting values with IFS
    for i in range(n):
        for j in range(n):
            if (i, j) in b_ij_start:
                b_ij[(i, j)].start = b_ij_start[(i, j)]
            else:
                b_ij[(i, j)].start = 0

    # model.setParam("MIPGap", 0.01)

    # Solve Model
    model.optimize()
    print()

    # END TIMER
    end = time()

    time_elapsed = end - start
    if model.status == GRB.OPTIMAL:
        assignments = {}
        for j in range(n):
            coms = []
            for i in range(n):
                if b_ij[(i, j)]:  ################## MODIFIED ####################
                    if b_ij[(i, j)].x != 0:
                        coms.append(i)
            if coms != []:
                assignments[j] = coms

        ls = []
        for i in range(n):
            for j in range(n):
                if b_ij[(i, j)].x == 1 and j not in ls:
                    ls.append(j)

        return f"IFS took {ifs_time} seconds. Total time: " + str(round(time_elapsed, 2)) + " seconds, Z : " + str(
            Z.x), ls, assignments
    else:
        return "Model Is Infeasible"


# print(sam_global_w_ifs2(10))



