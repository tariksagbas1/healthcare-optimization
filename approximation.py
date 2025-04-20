from gurobipy import Model, GRB, quicksum
from math import dist, sqrt
from numpy import percentile, linspace, array
from time import time
from sklearn.cluster import AgglomerativeClustering


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
            row.append(d_ij[i][j] * (p[i] + p[j]) / 2)

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


def sam_ifs1(dataset_index, units: list):
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

    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    b_ij = {}  # b_ij = 1 if community i is served by unit on j
    for unit_index in units:
        for i in range(n):
            b_ij[(i, unit_index)] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{unit_index}")

    # Z: auxillary variable
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

        for unit_index in units:
            for i in range(n):
                b_ij[(i, unit_index)] = b_ij[(i, unit_index)].x

        return b_ij, round(time_elapsed, 2)
    else:
        return "Model Is Infeasible"


def sam_apx3_w_ifs1(dataset_index, d_up = 90, p_lp = 10):

    # INITIAL DATA

    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()

    line1 = lines[0].split()
    n = int(line1[0]) # Number of nodes
    m = int(line1[1]) # Number of healthcare units to place

    for line in lines[2:]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, C, population = values[1], values[2], values[3], int(values[4])
        cord_com.append([x, y]) # add coordinates
        p.append(population)  # add populations for each community i

    # Calculate distances
    d_ij = []
    for x1, y1 in cord_com:
        row = []
        for x2, y2 in cord_com:
            distance = dist((x1, y1), (x2, y2))
            row.append(distance)
        d_ij.append(row)

    all_weighted_distances = [d_ij[i][j] * ((p[i] + p[j])/2) for i in range(n) for j in range(n) if i != j]
    d_max = percentile(all_weighted_distances, d_up)
    p_min = percentile(p, p_lp)

    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: if population i is served by unit on j
    b_ij = {}
    for i in range(n):
        for j in range(n):
            if d_ij[i][j] * ((p[i] + p[j])/2) < d_max:
                b_ij[(i, j)] = model.addVar(vtype = GRB.BINARY, name = f"b_{i}_{j}")
            else:
                b_ij[(i, j)] = None

    # z_j: 1 if facility is opened at node j. 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype = GRB.BINARY, name = f"z_{j}"))

    # Z: auxillary variable
    Z = model.addVar(vtype=GRB.CONTINUOUS, name="Z")

    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    # Z > b_ij * d_ij * p[i] for every i, j
    # Only add if p[i] isn't very small
    for i in range(n):
        variables = 0.0
        for j in range(n):
            if i != j and b_ij[(i, j)] and p[i] >= p_min:
                variables += p[i] * b_ij[i, j] * d_ij[i][j]
        model.addConstr(Z >= variables, name=f"Z_constraint_{j}")

    # Capacity constraint, sum(b_ij * p[i]) <= C for every unit. (surplus capacity allowed)
    for j in range(n):
        variables = 0.0
        for i in range(n):
            if b_ij[(i, j)]:
                variables += b_ij[(i, j)] * p[i]
        model.addConstr(variables <= C * z_j[j], name=f"capacity_constraint_{i}")

    # Population constraint
    for i in range(n):
        sum = 0.0
        for j in range(n):
            if b_ij[(i, j)]:
                sum += b_ij[(i, j)]
        model.addConstr(1 == sum, name=f"Population_constraint{i}")

    # Total unit constraint, sum(x_i) == m

    sum = 0.0
    for j in range(n):
        sum += z_j[j]
    model.addConstr(sum == m, name = f"unit_constraint")


    units = d_cluster(dataset_index)[0]
    b_ij_start, ifs_time = sam_ifs1(dataset_index, units)

    # Set starting values with IFS
    for i in range(n):
        for j in range(n):
            if b_ij[(i, j)]:
                if (i, j) in b_ij_start:
                    b_ij[(i, j)].start = b_ij_start[(i, j)]
                else:
                    b_ij[(i, j)].start = 0

    # START TIMER
    start = time()

    # Solve Model
    model.optimize()
    print()
    # END TIMER
    end = time()
    time_elapsed = end - start

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
            if b_ij[(i, j)]:
                if b_ij[(i, j)].x == 1 and j not in ls:
                    ls.append(j)
    return f"Optimality Gap: {model.MIPGap * 100:.2f}% " + str(round(time_elapsed, 2)) + " seconds, Z : " + str(
        Z.x), ls, assignments

# sam_apx3_w_ifs1(13, 60, 40)
