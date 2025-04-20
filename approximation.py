from gurobipy import Model, GRB, quicksum
from math import dist, sqrt
from numpy import percentile, linspace, array
from time import time
from matplotlib import pyplot
from sklearn.cluster import KMeans, AgglomerativeClustering
from maybe_lp_relax import lp_global, d_cluster, sam_ifs2

def approximate_ifs(dataset_index, d_up):
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

    all_distances = [d_ij[i][j] * ((p[i] + p[j]) / 2) for i in range(n) for j in range(n) if
                     i != j]  ################## MODIFIED ####################
    d_max = percentile(all_distances, d_up)  ################## MODIFIED ####################

    # units = [0, 14, 21, 25, 28, 31, 33, 35, 42, 46, 62, 63, 80, 86, 90, 92, 99, 106, 115, 122, 128, 142, 150, 153, 163, 168, 172, 184, 186, 189, 191, 195, 197]
    # cluster_time = 0

    units, cluster_time = d_cluster(dataset_index)
    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    b_ij = {}  # b_ij = 1 if community i is served by unit on j
    for unit_index in units:
        for i in range(n):
            if d_ij[i][unit_index] * ((p[i] + p[unit_index]) / 2) < d_max:
                b_ij[(i, unit_index)] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{unit_index}")
            else:
                b_ij[(i, unit_index)] = None

    # Z: auxiliary variable
    Z = model.addVar(vtype=GRB.CONTINUOUS, name="Z")

    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    # Z constraint
    for unit_index in units:
        for i in range(n):
            if i != unit_index and b_ij[(i, unit_index)]:
                model.addConstr(Z >= b_ij[(i, unit_index)] * d_ij[i][unit_index] * p[i],
                                name=f"Z_constraint_{i}_{unit_index}")

    # Single allocation constraint, sum(b_ij) == 1 for every population
    for i in range(n):
        variables = 0.0
        for unit_index in units:
            if b_ij[(i, unit_index)]:
                variables += b_ij[(i, unit_index)]
        model.addConstr(variables == 1, name=f"single_allocation_constraint_{i}")

    # Capacity constraint
    for unit_index in units:
        variables = 0.0
        for i in range(n):
            if b_ij[(i, unit_index)]:
                variables += b_ij[(i, unit_index)] * p[i]
        model.addConstr(variables <= C, name=f"capacity_constraint_{unit_index}")

    model.update()
    model.setParam(GRB.Param.MIPGap, 0.05)

    # START TIMER
    start = time()

    # Solve Model
    model.optimize()

    print()
    # model.write("ifs.sol")
    # END TIMER
    end = time()
    time_elapsed = end - start
    if model.status == GRB.OPTIMAL:
        z_val = Z.x
        for unit_index in units:
            for i in range(n):
                if b_ij[(i, unit_index)]:
                    b_ij[(i, unit_index)] = b_ij[(i, unit_index)].x
        tot_time = int(cluster_time) + round(time_elapsed, 2)

        return b_ij, tot_time , z_val
    else:
        return "Model Is Infeasible"

def approximation_w_ifs(dataset_index, d_up):
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

    all_distances = [d_ij[i][j] * ((p[i]+p[j]) / 2) for i in range(n) for j in range(n) if
                     i != j]  ################## MODIFIED ####################
    d_max = percentile(all_distances, d_up)  ################## MODIFIED ####################

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

    # APRROXIMATION METHOD
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

    # Z: auxiliary variable
    Z = model.addVar(vtype=GRB.CONTINUOUS, name="Z")

    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    # for j in range(n):
    #     model.addConstr(Z >= minimums[j] * (1 - z_j[j]))

    # model.addConstr(Z <= maximums[0][2])

    #######################    MODIFIED #############################################

    # Z > b_ij * d_ij for every i, j
    for i in range(n):
        model.addConstr(Z >= quicksum(p[i] * d_ij[i][j] * b_ij[i, j] for j in range(n) if i != j and b_ij[i, j]), name=f"Z_constraint_{i}")

    ##################################################################################

    # Capacity constraint, sum(b_ij * p[i]) <= C for every unit. (There can be surplus capacity ?)
    for j in range(n):
        variables = 0.0
        for i in range(n):
            if b_ij[(i, j)]:
                variables += b_ij[(i, j)] * p[i]
        model.addConstr(variables <= C * z_j[j], name=f"capacity_constraint_{j}")

    # Population constraint
    for i in range(n):
        sum = 0.0
        for j in range(n):
            if b_ij[(i, j)]:  ################## MODIFIED ####################
                sum += b_ij[(i, j)]
        model.addConstr(1 == sum, name=f"Population_constraint{i}")

    # Total unit constraint, sum(x_i) == m

    g_j = []
    for j in range(n):
        variables = 0.0
        for i in range(n):
            if b_ij[(i, j)]:  ################## MODIFIED ####################
                variables += b_ij[(i, j)]
        g_j.append(variables)
        model.addConstr(g_j[j] <= M * z_j[j])

    sum = 0.0
    for j in range(n):
        sum += z_j[j]
    model.addConstr(sum == m, name=f"unit_constraint")

    # START TIMER
    start = time()

    # SET STARTING VALUES FROM DIFFERENT SOLUTIONS

    cluster_index, ifs_time = d_cluster(dataset_index)
    # a, lp_index, b, lp_val = lp_global(dataset_index)

    # lp_set = set(lp_index)
    for j in cluster_index:     ################## MODIFIED ###########################
        z_j[j].start = 1

    # model.addConstr(Z >= lp_val)

    """
    b_ij_start, ifs_time, z_val = approximate_ifs(dataset_index, d_up)

    # Set starting values with IFS
    for i in range(n):
        for j in range(n):
            if b_ij[(i, j)]:
                if (i, j) in b_ij_start:
                    b_ij[(i, j)].start = b_ij_start[(i, j)]
                else:
                    b_ij[(i, j)].start = 0
    """

    # MODIFIED ############ PARAMETERS #####################################
    # model.setParam("Cutoff", z_val)
    # model.setParam("MIPGap", 0.2)
    # model.setParam("MIPFocus", 2)
    model.setParam("Heuristics", 0.3)
    # model.setParam("TimeLimit", 800)
    # idk anymore


    model.setParam("GomoryPasses", 5)
    model.setParam("FlowCoverCuts", 2)
    # model.setParam("CliqueCuts", 2)

    # Solve Model
    model.optimize()
    print()

    # END TIMER
    end = time()

    time_elapsed = end - start
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
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
        for j in range(n):
            if z_j[j].x == 1:
                ls.append(j)

        return f"IFS took {ifs_time} seconds. Total time: " + str(round(time_elapsed, 2)) + " seconds, Z : " + str(
                Z.x), ls, assignments
    else:
        return "Model Is Infeasible"

# a, tot_time, z_val = approximate_ifs(8, 90)
# print(tot_time, z_val)

# print(approximation_w_ifs(14, 80))
