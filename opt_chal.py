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

    M = len(units)
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

a = [2, 3, 4, 12, 13, 22, 23, 24, 28, 41, 58, 61, 64, 67, 73, 76, 83, 90, 94, 99, 105, 115, 121, 125, 127, 131, 134, 137, 143, 153, 154, 161, 164, 170, 181, 183, 190, 194, 199, 201, 212, 213, 220, 227, 248, 249, 252, 255, 259, 266, 279, 290, 298, 308, 309, 310, 311, 313, 315, 319, 337, 342, 349, 350, 351, 362, 379, 386, 392, 398]
b = {2: [112], 3: [66, 101, 297, 332], 4: [4, 88, 147, 230], 12: [45, 169, 182, 301, 305], 13: [44, 186, 248, 317, 357], 22: [22, 25, 368, 371], 23: [23, 26, 339], 24: [24, 165, 185, 236, 369], 28: [7, 28, 77, 109, 111], 41: [42, 96, 278, 343], 58: [59, 103, 123, 191, 240, 379], 61: [61, 114, 139, 196, 376], 64: [17, 39, 48, 60, 95, 187, 216, 261, 268, 273, 283, 330, 356], 67: [54, 73, 99, 260, 288, 378], 73: [72, 110, 154, 214, 238, 287, 360], 76: [76, 229, 262], 83: [83, 150, 193, 328], 90: [6, 11, 51, 149, 198, 285], 94: [33, 69, 94, 223, 394], 99: [18, 78, 140, 208, 380, 381], 105: [62, 71, 115, 157, 395], 115: [105, 126, 239, 244, 274, 370], 121: [85, 138, 148, 227, 281, 397], 125: [90, 125, 173, 222, 390], 127: [10, 40, 81, 233, 237], 131: [9, 56, 131, 171, 256, 327], 134: [12, 38, 74, 156, 242, 295, 338], 137: [68, 137, 225, 364, 367], 143: [50, 93, 122, 143, 352], 153: [1, 55, 132, 134, 177, 259, 271, 396], 154: [37, 167, 294, 377], 161: [142, 272, 316, 358], 164: [32, 141, 164, 329, 334, 341], 170: [161, 170, 200, 307, 383], 181: [0, 135, 152, 232, 303, 363], 183: [31, 41, 119, 178, 345], 190: [53, 79, 190, 269, 385], 194: [52, 82, 184, 194], 199: [106, 116, 199, 204, 325], 201: [19, 201, 254, 321, 344], 212: [8, 67, 97, 100, 205, 210, 212, 218, 246, 258, 275, 286, 322, 331], 213: [92, 189, 224, 347, 391, 398], 220: [124, 155, 179, 220, 291], 227: [121, 160, 215, 320, 336], 248: [13, 16, 304, 306], 249: [118, 162, 249, 277, 374], 252: [104, 117, 146, 209, 211], 255: [144, 175, 221, 250, 255, 359], 259: [86, 102, 163, 181, 353, 355], 266: [266, 299, 384, 389, 393], 279: [128, 133, 158, 267, 279, 296], 290: [15, 58, 290, 300, 375], 298: [166, 312, 362, 387], 308: [176, 308], 309: [36, 108, 136, 213, 231, 309, 361], 310: [46, 65, 129, 188, 247, 280, 292, 302, 310, 388], 311: [98, 226, 311, 326], 313: [5, 243, 284, 313, 333], 315: [63, 315], 319: [21, 234, 264, 319, 382], 337: [2, 113, 217, 219, 252, 253, 314, 324, 337], 342: [75, 80, 89, 235, 276, 342], 349: [192, 207, 349], 350: [183, 257, 293, 323, 350], 351: [20, 30, 49, 57, 91, 151, 153, 174, 197, 206, 282, 340, 348, 351, 373], 362: [159, 202, 203, 298], 379: [70, 130, 346, 354, 366], 386: [14, 27, 35, 43, 127, 145, 180, 228, 245, 263, 270, 289, 318, 335, 372, 386, 399], 392: [3, 34, 47, 64, 84, 87, 107, 120, 168, 172, 265, 365], 398: [29, 195, 241, 251, 392]}

print(cvrp_global(14, a, b))
