from gurobipy import Model, GRB, quicksum
from math import dist, sqrt
from numpy import percentile, linspace, array
from time import time
from sklearn.cluster import AgglomerativeClustering

# STAGE 1

# FINDING A WARM START

# "weighted_clustering" function uses Agglomerative Clustering to determine m clusters
# m facilities to be deployed
# returns decided facility indices and time elapsed
def weighted_clustering(dataset_index):
    # GET DATA AND CALCULATE NECESSARY PARAMETERS

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

    # Turn into weighted-distance matrix
    # weight approximation choice: geometric mean of populations

    w_d_ij = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(d_ij[i][j] * ((p[i] * p[j]) ** 0.5))

        w_d_ij.append(row)

    start = time()

    # Fit clustering on weighted-distance matrix
    import numpy as np
    w_d_ij = np.array(w_d_ij)
    clustering = AgglomerativeClustering(n_clusters=m, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(w_d_ij)

    # Group node indices by cluster
    clusters = [[] for _ in range(m)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    # Find the central node for each cluster
    central_nodes = []

    for group in clusters:
        submatrix = w_d_ij[np.ix_(group, group)]  # cluster's intra-distance matrix
        total_dist = submatrix.sum(axis=1)  # total distance from each node to others
        best_idx_in_cluster = group[np.argmin(total_dist)]
        central_nodes.append(best_idx_in_cluster)
    end = time()
    return central_nodes, round(end - start, 2)

# "ifs_by_clustering" function uses the nodes selected in "weighted_clustering" and
# solves for it
# returns b_ij binary variables, time elapsed and objective value
# this function is used as the solver for datasets with more than 1000 nodes.
def ifs_by_clusters(dataset_index):
    # DATA

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

    # Get unit indices from clustering
    units, cluster_time = weighted_clustering(dataset_index)

    # Create model

    model = Model("Healthcare Placement")

    # DECISION VARIABLES

    b_ij = {}  # b_ij = 1 if community i is served by unit on j
    for unit_index in units:
        for i in range(n):
            b_ij[(i, unit_index)] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{unit_index}")

    # Z: auxiliary variable
    Z = model.addVar(vtype=GRB.CONTINUOUS, name="Z")

    model.update()

    # OBJECTIVE FUNCTION

    model.setObjective(Z, GRB.MINIMIZE)

    # CONSTRAINTS

    # Z constraints
    for unit_index in units:
        for i in range(n):
            if i != unit_index:
                model.addConstr(Z >= b_ij[(i, unit_index)] * d_ij[i][unit_index] * p[i],
                                name=f"Z_constraint_{i}_{unit_index}")

    # A population can only be served by one facility
    for i in range(n):
        variables = 0.0
        for unit_index in units:
            variables += b_ij[(i, unit_index)]
        model.addConstr(variables == 1, name=f"single_allocation_constraint_{i}")

    # Capacity constraints
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

# solves partially relaxed problem, finds global optimum
# b_ij: BINARY -> CONTINUOUS, lb=0, ub=1
def lp_global(dataset_index):
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
            b_ij[(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb= 0, ub= 1, name=f"b_{i}_{j}")

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

    minimums.sort(reverse=True)
    # model.addConstr(Z >= minimums[n//10])
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

    # A node can only serve if a facility is opened there
    g_j = []
    for j in range(n):
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, j)]
        g_j.append(variables)
        model.addConstr(g_j[j] <= M * z_j[j])

    # Total unit constraint, sum(z_j) == m
    sum = 0.0
    for j in range(n):
        sum += z_j[j]
    model.addConstr(sum == m, name=f"unit_constraint")

    # START TIMER
    start = time()

    # Solve Model
    model.optimize()

    print()
    # END TIMER
    end = time()
    time_elapsed = end - start
    if model.status == GRB.OPTIMAL:
        # get the solution
        assignments = {}
        for j in range(n):
            coms = []
            for i in range(n):
                if b_ij[(i, j)].x != 0:
                    coms.append(i)
            if coms:
                assignments[j] = coms

        ls = []
        for j in range(n):
            if z_j[j].x == 1:
                ls.append(j)
        z_val = Z.x
        return round(time_elapsed, 2), " seconds, Z : " , Z.x, ls, assignments
    else:
        return "Model Is Infeasible"

# SOLVING

def solver_w_ifs(dataset_index):
    # DATA

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

    # Calculate distances
    d_ij = []
    for x1, y1 in cord_com:
        row = []
        for x2, y2 in cord_com:
            distance = dist((x1, y1), (x2, y2))
            row.append(distance)
        d_ij.append(row)

    # Create lists "minimums" and "maximums"
    # that hold minimum weighted distance to node i at element i
    # and maximum weighted distance to node i at element i, respectively
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

    # b_ij: 1 if population i is served by unit on j, 0 o/w
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}")

    # z_j: 1 if a facility is opened at node j, 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype=GRB.BINARY, name=f"z_{j}"))

    # Z: auxiliary variable
    Z = model.addVar(vtype=GRB.CONTINUOUS, name="Z")

    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    # Z must be larger than the maximum of the minimum weighted distances to nodes
    # since every node must be served
    for j in range(n):
        model.addConstr(Z >= minimums[j] * (1 - z_j[j]))

    # Z is smaller than the maximum population weighted distance possibly served
    model.addConstr(Z <= maximums[0][2])


    # Z > sum(p[i] * b_ij * d_ij for every j) for every i
    for i in range(n):
        model.addConstr(Z >= quicksum(p[i] * d_ij[i][j] * b_ij[i, j] for j in range(n) if i != j), name=f"Z_constraint_{i}")


    # Capacity constraint, sum(b_ij * p[i]) <= C * z_j for every unit. (surplus capacity allowed)
    for j in range(n):
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, j)] * p[i]
        model.addConstr(variables <= C * z_j[j], name=f"capacity_constraint_{j}")

    # Population constraint
    for i in range(n):
        sum = 0.0
        for j in range(n):
            sum += b_ij[(i, j)]
        model.addConstr(1 == sum, name=f"Population_constraint{i}")

    # A node can only serve if a facility is opened there.
    # using big M
    g_j = []
    for j in range(n):
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, j)]
        g_j.append(variables)
        model.addConstr(g_j[j] <= M * z_j[j])

    # Total unit constraint, sum(z_j) == m
    sum = 0.0
    for j in range(n):
        sum += z_j[j]
    model.addConstr(sum == m, name=f"unit_constraint")

    # START TIMER
    start = time()

    # Get starting facility locations

    cluster_index = weighted_clustering(dataset_index)[0]

    for j in cluster_index:
        z_j[j].start = 1

    # Set LP objective value as lowerbound
    # LP is sometimes not the worth the time
    a, b, lp_val, lp_units, d = lp_global(dataset_index)
    model.addConstr(Z >= lp_val)

    # Get starting solution
    b_ij_start, ifs_time, z_val = ifs_by_clusters(dataset_index)

    # Set starting values with IFS
    for i in range(n):
        for j in range(n):
            if (i, j) in b_ij_start:
                b_ij[(i, j)].start = b_ij_start[(i, j)]
            else:
                b_ij[(i, j)].start = 0

    # DEFAULT PARAMETERS
    # Modify if necessary, some datasets are solved better/faster with different parameters

    model.setParam("Heuristics", 0.3) # helps tighten the upper bound faster
    model.setParam("TimeLimit", 800) # get a good enough solution in reasonable amount of time
    model.setParam("NumericFocus", 1) # makes gurobi more careful with numbers,
    # good if values are too close to each other
    model.setParam("Cutoff", z_val * (1 + 0.0001)) # prune any solution worse than ifs objective value
    model.setParam("GomoryPasses", 5) # better cuts to enhance bounds

    # Solve Model
    model.optimize()
    print()

    # END TIMER
    end = time()

    time_elapsed = end - start
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # get the solution
        assignments = {}
        for j in range(n):
            coms = []
            for i in range(n):
                if b_ij[(i, j)].x > 0.5:
                    coms.append(i)
            if coms:
                assignments[j] = coms

        ls = []
        for j in range(n):
            if z_j[j].x == 1:
                ls.append(j)

        return f"IFS took {ifs_time} seconds. Total time: " + str(round(time_elapsed, 2)) + " seconds, Z : ", Z.x, ls, assignments
    else:
        return "Model Is Infeasible"

# SOLVE MORE COMPLEX DATASETS: PHASES WITH DIFFERENT PARAMETERS
# Phase 1: tries to find the best feasible solution it can find
# stops at time limit and writes the current best solution to a file "phase1.sol"
def phase_1(dataset_index):
    # DATA

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

    # Calculate distances
    d_ij = []
    for x1, y1 in cord_com:
        row = []
        for x2, y2 in cord_com:
            distance = dist((x1, y1), (x2, y2))
            row.append(distance)
        d_ij.append(row)

    # Create lists "minimums" and "maximums"
    # that hold minimum weighted distance to node i at element i
    # and maximum weighted distance to node i at element i, respectively
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

    # DECISION VARIABLES

    # b_ij: 1 if node i is served by facility at node j, 0 o/w
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}")

    # z_j: 1 if facility is opened at node j, 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype=GRB.BINARY, name=f"z_{j}"))

    # Z: auxiliary variable
    Z = model.addVar(vtype=GRB.CONTINUOUS, name="Z")

    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # CONSTRAINTS

    # Z > (p_i * b_ij * d_ij sum over all j) for every i
    for i in range(n):
        model.addConstr(Z >= quicksum(p[i] * d_ij[i][j] * b_ij[i, j] for j in range(n)), name=f"Z_constraint_{i}")

    # Z must be larger than the maximum of the minimum weighted distances to nodes
    # since every node must be served
    for j in range(n):
        model.addConstr(Z >= minimums[j] * (1 - z_j[j]))

    # Z is smaller than the largest population weighted distance possibly served
    model.addConstr(Z <= maximums[0][2])

    # Capacity constraint, sum(b_ij * p[i]) <= C * z_j for every unit. (surplus capacity allowed)
    for j in range(n):
        variables = quicksum(b_ij[(i, j)] * p[i] for i in range(n))
        model.addConstr(variables <= C * z_j[j], name=f"capacity_constraint_{i}")

    # Population constraint
    for i in range(n):
        exprs = quicksum(b_ij[(i, j)] for j in range(n))
        model.addConstr(exprs == 1, name=f"Population_constraint{i}")

    # A node can only serve if a facility is opened there
    for j in range(n):
        g_j = quicksum(b_ij[(i, j)] for i in range(n))
        model.addConstr(g_j <= M * z_j[j])

    # Total unit constraint, sum(z_j) == m
    tot = quicksum(z_j[j] for j in range(n))
    model.addConstr(tot == m, name=f"unit_constraint")

    # START TIMER
    start = time()

    # Get starting facility locations
    cluster_index, cluster_time = weighted_clustering(dataset_index)
    for j in cluster_index:
        z_j[j].start = 1

    # Get starting solution
    b_ij_start, ifs_time, z_val = ifs_by_clusters(dataset_index)

    # Set starting values with IFS
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)].Start = b_ij_start.get((i, j), 0)

    # DEFAULT PARAMETERS
    # Modify if necessary, some datasets are solved better/faster with different parameters

    model.setParam("Cutoff", z_val * (1 + 0.0001)) # prune any solution worse than ifs objective value
    model.setParam("TimeLimit", 300) # get a good enough solution in reasonable amount of time
    model.setParam("NumericFocus", 1) # makes gurobi more careful with numbers,
    # good if values are too close to each other
    model.setParam("Heuristics", 0.3) # helps tighten the upper bound faster

    # OPTIONAL Parameters:
    # better cuts to enhance bounds, may slow down the process if bounds are already good
    model.setParam("GomoryPasses", 5)
    # model.setParam("FlowCoverCuts", 2)
    # model.setParam("MIPFocus", 1)  # focus on finding feasible solutions fast, lower the upper bound,
    # useful where global optimum takes an infeasible amount of time to find
    # model.setParam("ConcurrentMIP", 1) # use different methods at the same time
    # by assigning CPU cores to different jobs, take the best bound found in any and continue

    # Solve Model
    model.optimize()
    print()

    # END TIMER
    end = time()

    time_elapsed = end - start
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # get the solution
        assignments = {}
        for j in range(n):
            coms = []
            for i in range(n):
                if b_ij[(i, j)].x > 0.5:
                    coms.append(i)
            if coms:
                assignments[j] = coms

        ls = []
        for j in range(n):
            if z_j[j].x > 0.5:
                ls.append(j)

        # store the solution
        model.write("phase1.sol")

        return f"IFS took {ifs_time + cluster_time} seconds. Total time: " + str(round(time_elapsed, 2)) + " seconds, Z : ", \
            Z.x, "Best bound: ", model.ObjBound, ls, assignments
    else:
        return "Model Is Infeasible"

# Phase 2: reads ifs from "phase1.sol"
# and tries to get close to optimality as much as possible
# stops at time limit and returns the best solution found
def phase_2(dataset_index, best_bound, phase1_val):
    # DATA

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

    # Calculate distances
    d_ij = []
    for x1, y1 in cord_com:
        row = []
        for x2, y2 in cord_com:
            distance = dist((x1, y1), (x2, y2))
            row.append(distance)
        d_ij.append(row)

    # Create lists "minimums" and "maximums"
    # that hold minimum weighted distance to node i at element i
    # and maximum weighted distance to node i at element i, respectively
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

    # DECISION VARIABLES

    # b_ij: 1 if node i is served by facility at node j, 0 o/w
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}")

    # z_j: 1 if facility is opened at node j, 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype=GRB.BINARY, name=f"z_{j}"))

    # Z: auxiliary variable
    Z = model.addVar(vtype=GRB.CONTINUOUS, name="Z")

    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # CONSTRAINTS

    # Z > (p_i * b_ij * d_ij sum over all j) for every i
    for i in range(n):
        model.addConstr(Z >= quicksum(p[i] * d_ij[i][j] * b_ij[i, j] for j in range(n)), name=f"Z_constraint_{i}")

    # Z must be larger than the maximum of the minimum weighted distances to nodes
    # since every node must be served
    for j in range(n):
        model.addConstr(Z >= minimums[j] * (1 - z_j[j]))

    # Z is smaller than the largest population weighted distance possibly served
    model.addConstr(Z <= maximums[0][2])


    # Capacity constraint, sum(b_ij * p[i]) <= C * z_j for every unit. (surplus capacity allowed)
    for j in range(n):
        variables = quicksum(b_ij[(i, j)] * p[i] for i in range(n))
        model.addConstr(variables <= C * z_j[j], name=f"capacity_constraint_{i}")

    # Population constraint
    for i in range(n):
        exprs = quicksum(b_ij[(i, j)] for j in range(n))
        model.addConstr(exprs == 1, name=f"Population_constraint{i}")

    # A node can only serve if a facility is opened there
    for j in range(n):
        g_j = quicksum(b_ij[(i, j)] for i in range(n))
        model.addConstr(g_j <= M * z_j[j])


    # Total unit constraint, sum(z_j) == m
    tot = quicksum(z_j[j] for j in range(n))
    model.addConstr(tot == m, name=f"unit_constraint")

    # Set the phase 1 best bound as lower bound
    model.addConstr(Z >= best_bound)

    # START TIMER
    start = time()

    # Get starting solution
    model.read("phase1.sol")

    # DEFAULT PARAMETERS
    # Modify if necessary, some datasets are solved better/faster with different parameters

    model.setParam("Cutoff", phase1_val * (1 + 0.0001)) # prune any solution worse than phase 1 objective value
    model.setParam("NumericFocus", 1) # makes gurobi more careful with numbers,
    # good if values are too close to each other
    model.setParam("GomoryPasses", 5) # better cuts to enhance bounds
    model.setParam('Presolve', 2) # tells gurobi to do a better presolve
    model.setParam("Heuristics", 0.3) # helps tighten the upper bound faster
    model.setParam("TimeLimit", 800) # get a good enough solution in reasonable amount of time
    # also useful when gurobi already found the global optimum but is trying to prove optimality
    # and stalls at a certain optimality gap

    # Solve Model
    model.optimize()
    print()

    # END TIMER
    end = time()

    time_elapsed = end - start
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # get the solution
        assignments = {}
        for j in range(n):
            coms = []
            for i in range(n):
                if b_ij[(i, j)].x != 0:
                    coms.append(i)
            if coms:
                assignments[j] = coms

        ls = []
        for j in range(n):
            if z_j[j].x == 1:
                ls.append(j)

        return ls, Z.x, str(round(time_elapsed, 2)) + " seconds", assignments
    else:
        return "Model Is Infeasible"

# STAGE 2

print(phase_1(14))
# print(phase_2(10, 16328.87594912057, 22652.355815676212))
