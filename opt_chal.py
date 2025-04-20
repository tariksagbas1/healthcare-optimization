from gurobipy import Model, GRB, quicksum
from math import dist, sqrt
from matplotlib import pyplot
from numpy import percentile, linspace, array
from time import time
from sklearn.cluster import AgglomerativeClustering

# PLOTTING

def plot_function(dataset_index, units=[], assignments={}):
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file:  # Read from file
        lines = file.readlines()
    p = []
    cord_com = []
    i = 0
    community_names = []
    for line in lines[2:]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, population = values[1], values[2], int(values[4])
        cord_com.append((x, y))  # add coordinates
        p.append(population)  # add populations for each community i
        community_names.append(i)
        i += 1

    colors = ['red' if name in units else 'black' for name in community_names]
    x, y = zip(*cord_com)
    pyplot.scatter(x, y, color=colors)
    pyplot.grid(True)

    for i, name in enumerate(community_names):
        pyplot.annotate(name, (x[i], y[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    for unit, served_communities in assignments.items():  # assuming this is your dictionary
        unit_x, unit_y = cord_com[unit]
        for comm in served_communities:
            comm_x, comm_y = cord_com[comm]
            pyplot.plot([unit_x, comm_x], [unit_y, comm_y], linestyle='--', color='gray', linewidth=0.5)

    pyplot.xticks(linspace(min(x), max(x), num=20))  # 20 intervals along the x-axis
    pyplot.yticks(linspace(min(y), max(y), num=20))  # 20 intervals along the y-axis
    pyplot.show()

def plot_function2(dataset_index, test_units = [], true_units = [], assignments = {}):
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()
    p = []
    cord_com = []
    i = 0
    community_names = []
    for line in lines[2:]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, population = values[1], values[2], int(values[4])
        cord_com.append((x, y)) # add coordinates
        p.append(population)  # add populations for each community i
        community_names.append(i)
        i += 1

    colors = [
    'magenta' if name in true_units and name in test_units
    else 'red' if name in true_units
    else 'green' if name in test_units
    else 'black'
    for name in community_names
    ]
    x, y = zip(*cord_com)
    pyplot.scatter(x, y, color=colors)
    pyplot.grid(True)

    for i, name in enumerate(community_names):
        pyplot.annotate(name, (x[i], y[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    for unit, served_communities in assignments.items():  # assuming this is your dictionary
        unit_x, unit_y = cord_com[unit]
        for comm in served_communities:
            comm_x, comm_y = cord_com[comm]
            pyplot.plot([unit_x, comm_x], [unit_y, comm_y], linestyle='--', color='gray', linewidth=0.5)

    pyplot.xticks(linspace(min(x), max(x), num=20))  # 20 intervals along the x-axis
    pyplot.yticks(linspace(min(y), max(y), num=20))  # 20 intervals along the y-axis
    pyplot.show()

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

    model.addConstr(Z <= maximums[0][2])

    # Z > p_i * b_ij * d_ij for every i, j
    for i in range(n):
        for j in range(n):
            model.addConstr(Z >= p[i] * b_ij[i, j] * d_ij[i][j], name=f"Z_constraint_{i}_{j}")

    # Capacity constraint, sum(b_ij * p[i]) <= C * z_j for every unit. (surplus capacity allowed)
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
                if b_ij[(i, j)].x > 0.5:
                    coms.append(i)
            if coms:
                assignments[j] = coms

        ls = []
        for j in range(n):
            if z_j[j].x > 0.5:
                ls.append(j)

        return str(round(time_elapsed, 2)), " seconds, Z : " , Z.x, ls, assignments
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

# SOLVE MORE COMPLEX DATASETS, METHOD 1:
# PHASES

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
    model.setParam("MIPFocus", 1)  # focus on finding feasible solutions fast, lower the upper bound,
    # useful where global optimum takes an infeasible amount of time to find
    model.setParam("ConcurrentMIP", 1) # tells gurobi to use different methods at the same time
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
    model.setParam("FlowCoverCuts", 2)
    model.setParam("MIPFocus", 1) # optional,
    # can be used to find better solutions if proving optimality takes an infeasible amount of time
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

# SOLVE MORE COMPLEX DATASETS, METHOD 2:
# APPROXIMATION BY DOWNSCALING

# "sam_apx3_w_ifs1" function reduces the number of variables and constraints
# by excluding too large weighted distances and too little populations

def sam_apx3_w_ifs1(dataset_index, d_up=90, p_lp=10):
    # INITIAL DATA

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

    all_weighted_distances = [d_ij[i][j] * ((p[i] + p[j]) / 2) for i in range(n) for j in range(n) if i != j]
    d_max = percentile(all_weighted_distances, d_up)
    p_min = percentile(p, p_lp)

    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: if population i is served by unit on j
    b_ij = {}
    for i in range(n):
        for j in range(n):
            if d_ij[i][j] * ((p[i] + p[j]) / 2) < d_max:
                b_ij[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}")
            else:
                b_ij[(i, j)] = None

    # z_j: 1 if facility is opened at node j. 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype=GRB.BINARY, name=f"z_{j}"))

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
    model.addConstr(sum == m, name=f"unit_constraint")


    b_ij_start, ifs_time, obj_val = ifs_by_clusters(dataset_index)

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


# STAGE 2

def cvrp_global(dataset_index, units=list(), assignments=dict()):
    full_units = [0] + units
    unit_cords = []  # Healthcare unit coordinates
    q = {}  # Unit equipment need
    p = []  # Population of every community

    M = 4
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file:
        lines = file.readlines()

    line2 = lines[1].split()
    depot_cord = (float(line2[1]), float(line2[2]))  # Depot coordinates
    unit_cords.append(depot_cord)

    for line in lines[2:]:
        vals = list(map(float, line.split()))
        index = vals[0]
        p.append(vals[4])
        if index - 1 in units:
            cords = (vals[1], vals[2])
            unit_cords.append(cords)

    for unit in assignments:
        total = 0
        for com in assignments[unit]:
            total += p[com]
        q[unit] = total

    d_ij = {}
    for i, (x1, y1) in enumerate(unit_cords):
        for j, (x2, y2) in enumerate(unit_cords):
            distance = dist((x1, y1), (x2, y2))
            index1 = full_units[i]
            index2 = full_units[j]
            d_ij[(index1, index2)] = distance

    model = Model("Ambulance Routing")

    # Decision variables
    x_ijk = {}
    for i in full_units:
        for j in full_units:
            for k in range(M):
                x_ijk[(i, j, k)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}{j}{k}")

    # Auxillary variable
    u_i = {}
    for i in units:
        u_i[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=q[i], ub=10000)

    # 1 if ambulance k is deployed, 0 otherwise
    y_k = {}
    for k in range(M):
        y_k[k] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}")

    Z = 0.0
    for i in full_units:
        for j in full_units:
            for k in range(M):
                Z += x_ijk[(i, j, k)] * d_ij[(i, j)]

    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    # A node is visited by a single ambulance

    for i in units:
        total = 0.0
        for k in range(M):
            for j in full_units:
                if i != j:
                    total += x_ijk[(i, j, k)]
        model.addConstr(total == 1, name=f"single_vehicle_constraint_{i}_{k}")

    total = 0.0
    for k in range(M):
        for j in full_units:
            if 0 != j:
                total += x_ijk[(0, j, k)]
    model.addConstr(total <= M, name=f"single_vehicle_constraint_{i}_{k}")

    for k in range(M):
        total = 0
        for i in units:
            for j in units:
                if i != j:
                    total += x_ijk[(i, j, k)] * q[j]
        model.addConstr(total <= 10000 * y_k[k], name=f"equipment_constraint_{k}")

    # Must include starting 0 node
    for k in range(M):
        total = 0.0
        for j in units:
            total += x_ijk[(0, j, k)]
        model.addConstr(total == 1 * y_k[k], name=f"starting_node_constraint_{k}")

        # Must include ending 0 node
    for k in range(M):
        total = 0.0
        for i in units:
            total += x_ijk[(i, 0, k)]
        model.addConstr(total == 1 * y_k[k], name=f"ending_node_constraint_{k}")

        # No going back to the same node you came from
    total = 0.0
    for k in range(M):
        for i in full_units:
            total += x_ijk[(i, i, k)]
    model.addConstr(total == 0, name="No_self_loops_constraint")

    # Flow constraint
    for i in units:
        for k in range(M):
            outflow = 0.0
            inflow = 0.0
            for j in full_units:
                if i != j:
                    outflow += x_ijk[(i, j, k)]
                    inflow += x_ijk[(j, i, k)]
            model.addConstr(outflow == inflow, name=f"flow_constraint_{i}_{k}")

    # Ambulance count constraint
    total = 0.0
    for k in range(M):
        total += y_k[k]
    model.addConstr(total <= len(units))

    # Subtour elimination constraint
    for i in units:
        for j in units:
            if i != j:
                for k in range(M):
                    model.addConstr(u_i[i] - u_i[j] + (10000 * x_ijk[(i, j, k)]) <= 10000 - q[j],
                                    name=f"SEC_{(i, j, k)}")

    model.optimize()
    if True:

        routes = {}
        route_count = 1

        for k in range(M):
            # Collect all active arcs for ambulance k
            arc_list = []
            for i in full_units:
                for j in full_units:
                    if i != j and x_ijk[(i, j, k)].x > 0.5:
                        arc_list.append((i, j))

            if not arc_list:
                continue  # skip unused ambulances

            # Build ordered route starting from depot
            route = [0]
            current = 0
            while True:
                found = False
                for (i, j) in arc_list:
                    if i == current:
                        route.append(j)
                        current = j
                        arc_list.remove((i, j))
                        found = True
                        break
                if not found or current == 0:
                    break
            routes[f"Route_{route_count}"] = route
            route_count += 1
        # Now print or return it
        return model.ObjVal, routes

a = [3, 5, 6, 9, 14, 18, 25, 32, 37, 58, 60, 68, 69, 78, 81, 99, 103, 118, 119, 122, 128, 134, 145, 147, 152, 153, 159, 170, 185, 187, 190, 191, 195, 199, 205, 209, 212, 216, 220, 221, 234, 240, 243, 250, 258, 270, 275, 280, 281, 283, 284, 289, 293, 308, 309, 311, 321, 344, 360, 362, 363, 370, 379, 383, 399]
b = {3: [3, 6, 35, 47, 93, 96, 358], 5: [77, 127, 145, 300, 351], 6: [48, 104, 138, 245, 287, 299, 335], 9: [61, 83, 269, 281, 307], 14: [11, 72, 201, 256, 272, 312], 18: [18, 22, 176, 211, 288, 326], 25: [25, 289, 305, 364], 32: [32, 178, 317, 378, 398], 37: [92, 120, 271, 366, 384], 58: [58, 113, 167, 184, 333], 60: [49, 60, 75, 298, 388], 68: [196, 202, 352, 367, 397], 69: [69, 78, 95, 161, 237, 295], 78: [26, 28, 37, 157, 191, 267, 368], 81: [33, 53, 107, 110, 114, 170, 233, 339, 342, 359], 99: [152, 185, 225, 227, 263], 103: [13, 64, 163, 193, 215, 261, 314], 118: [118, 124, 130, 190, 197], 119: [165, 210, 219, 265, 304, 310, 334], 122: [4, 10, 45, 291, 341, 346, 377], 128: [115, 128, 149, 179, 217, 239], 134: [21, 97, 103, 154, 160, 203, 222, 231, 259], 145: [20, 141, 162, 174, 242, 297, 350, 389], 147: [38, 94, 144, 147, 228, 230], 152: [30, 88, 99, 224, 253, 268, 387], 153: [31, 70, 153, 254, 353], 159: [1, 14, 146, 290, 372], 170: [44, 89, 164, 234, 235, 323, 327], 185: [9, 91, 142, 173, 181, 255, 276], 187: [8, 192, 220, 274, 348, 379], 190: [232, 252, 294, 337, 345, 381], 191: [34, 129, 143, 248, 258], 195: [41, 98, 140, 177, 262, 370], 199: [42, 59, 331, 394], 205: [100, 139, 155, 187, 283, 313, 349], 209: [137, 175, 277, 338, 374], 212: [131, 207, 208, 223, 229, 279, 325, 365], 216: [5, 46, 66, 73, 80, 116, 123, 156, 216, 244, 306, 316, 332, 336, 344, 392], 220: [63, 105, 125, 205, 249, 257, 282, 347], 221: [62, 194, 221, 251, 275, 354], 234: [108, 150, 151, 166, 296, 396], 240: [102, 195, 204, 226, 260], 243: [17, 39, 243, 246, 293, 319, 380], 250: [23, 24, 27, 121, 320], 258: [52, 76, 132, 159, 206, 395], 270: [57, 85, 266, 270, 357], 275: [51, 199, 315, 324], 280: [12, 79, 111, 133, 280], 281: [2, 65, 82, 87, 213, 399], 283: [50, 55, 158, 172, 302, 303], 284: [122, 168, 183, 284, 375], 289: [81, 86, 106, 188, 301, 308, 322], 293: [7, 16, 56, 182, 240, 285], 308: [36, 54, 74, 369], 309: [67, 119, 169, 189, 236, 247, 264, 328, 356, 376], 311: [180, 238, 373, 393], 321: [90, 135, 214, 218, 321, 371], 344: [0, 212, 250, 340, 391], 360: [198, 209, 318, 360], 362: [126, 273, 278, 362, 390], 363: [19, 40, 329, 355, 361], 370: [112, 200, 286, 330, 382, 385, 386], 379: [29, 71, 84, 136, 171, 241, 363], 383: [15, 68, 101, 117, 134, 148, 383], 399: [43, 109, 186, 292, 309, 311, 343]}

print(cvrp_global(13, a, b))
