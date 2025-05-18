from gurobipy import Model, GRB, quicksum
from math import dist, sqrt
from numpy import percentile, linspace, array
from time import time
from matplotlib import pyplot
from sklearn.cluster import AgglomerativeClustering

"""
PLOTTING
"""
def plot_function(dataset_index, units = [], assignments = {}):
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

    colors = ['red' if name in units else 'black' for name in community_names]
    x, y = zip(*cord_com)
    pyplot.scatter(x, y, color=colors)
    pyplot.grid(True)

    
    for i, name in enumerate(community_names):
        pyplot.annotate(name, (x[i], y[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    
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
        pyplot.annotate(name, (x[i], y[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    
    for unit, served_communities in assignments.items():  # assuming this is your dictionary
        unit_x, unit_y = cord_com[unit]
        for comm in served_communities:
            comm_x, comm_y = cord_com[comm]
            pyplot.plot([unit_x, comm_x], [unit_y, comm_y], linestyle='--', color='gray', linewidth=0.5)

    
    pyplot.xticks(linspace(min(x), max(x), num=20))  # 20 intervals along the x-axis
    pyplot.yticks(linspace(min(y), max(y), num=20))  # 20 intervals along the y-axis    
    pyplot.show()
"""
CLUSTERING
"""
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
"""
NEW CLUSTERING
"""
def d_cluster_new(dataset_index):
    # Initial data (unchanged)
    p = []  # Population of every community
    cord_com = []  # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: 
        # Read from file
        lines = file.readlines()

        line1 = lines[0].split()
        n = int(line1[0])  # Number of nodes
        m = int(line1[1])  # Number of healthcare units to place

        for line in lines[2:]:  # skip first two lines
            values = list(map(float, line.split()))  # Convert all values to
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

    #tresholds
    total_demand = sum(p)
    alpha = round(0.2 * (total_demand / m))
    max_dist = max(max(row) for row in d_ij)
    beta = 0.2 * max_dist


    start = time()

    # Step 1: Fit clustering on distance matrix
    import numpy as np
    w_d_ij = np.array(w_d_ij)
    clustering = AgglomerativeClustering(n_clusters= m, metric =
    'precomputed', linkage='average')
    labels = clustering.fit_predict(w_d_ij)

    # Step 2: Group node indices by cluster
    clusters = [[] for _ in range(m)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    # Step 3: For each cluster, find the most "central" node
    # (the one with minimum total distance to all others in its cluster)
    closest_nodes = []

    for group in clusters:
        submatrix = w_d_ij[np.ix_(group, group)]
        total_dist = submatrix.sum(axis=1)  # total distance from each
        best_idx_in_cluster = group[np.argmin(total_dist)]
        closest_nodes.append(best_idx_in_cluster)


    # Calculate workloads
    cluster_pops = [sum(p[i] for i in group) for group in clusters]
    workload_diff = max(cluster_pops) - min(cluster_pops)

    #Calculate access distances
    access_dists = [0] * n
    for c, group in enumerate(clusters):
        facility = closest_nodes[c]
        for i in group:
            access_dists[i] = d_ij[i][facility]
    access_diff = max(access_dists) - min(access_dists)

    # Heuristic repair if α or β violated
    max_iter = 50
    iter_count = 0
    while (workload_diff > alpha or access_diff > beta) and iter_count < max_iter:
        moved = False
        for c, group in enumerate(clusters):
            facility = closest_nodes[c]
            for i in group[:]:
                if i == facility:
                    continue  # skip the central node

                # Distance to other cluster centers
                other_centers = [closest_nodes[k] for k in range(m) if k != c]
                distances_to_others = [w_d_ij[i][f] for f in other_centers]
                nearest_c_idx = np.argmin(distances_to_others)
                target_cluster_idx = [k for k in range(m) if k != c][nearest_c_idx]

                # Try moving node to another cluster
                if p[i] + sum(p[j] for j in clusters[target_cluster_idx]) <= (total_demand / m + alpha):
                    clusters[c].remove(i)
                    clusters[target_cluster_idx].append(i)
                    moved = True
                    break
            if moved:
                break

        if not moved:
            break  # no valid move found

        # Recalculate centers and metrics
        closest_nodes = []
        for group in clusters:
            submatrix = w_d_ij[np.ix_(group, group)]
            best_idx = group[np.argmin(submatrix.sum(axis=1))]
            closest_nodes.append(best_idx)

        cluster_pops = [sum(p[i] for i in group) for group in clusters]
        workload_diff = max(cluster_pops) - min(cluster_pops)

        access_dists = [0] * n
        for c, group in enumerate(clusters):
            facility = closest_nodes[c]
            for i in group:
                access_dists[i] = d_ij[i][facility]
        access_diff = max(access_dists) - min(access_dists)

        iter_count += 1

    end = time()
    return closest_nodes, round(end - start, 2)

"""
SINGLE ALLOCATION MODELS
 - Simple global optimum model
"""
def sam_global(dataset_index):

    # INITIAL DATA

    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    line1 = lines[0].split()
    n = int(line1[0]) # Number of
    m = int(line1[1]) # Number of healthcare units to place
    M = n

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

    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: 1 if population i is served by unit on j. 0 otherwise
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype = GRB.BINARY, name = f"b_{i}_{j}")

    # z_j: 1 if facility is opened at node j. 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype = GRB.BINARY, name = f"z_{j}"))

    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 
    
    model.update()

    model.setObjective(Z, GRB.MINIMIZE)
    
    model.setParam("MIPFocus", 3)
    model.setParam("Heuristics", 0.5)
    model.setParam("Cuts", 2)
    model.setParam("VarBranch", 1)
    model.setParam("NodefileStart", 0.5)
    # Constraints
    
    # Z > p[i] * b_ij * d_ij for every i, j
    for i in range(n):
        for j in range(n):
            model.addConstr(Z >= p[i] * b_ij[i, j] * d_ij[i][j], name = f"Z_constraint_{i}_{j}")
    
    # Capacity constraint, sum(b_ij * p[i]) <= C for every unit. (There can be surplus capacity ?)
    for j in range(n):
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, j)] * p[i]
        model.addConstr(variables <= C * z_j[j], name = f"capacity_constraint_{j}")
  
    # Population constraint
    for i in range(n):
        sum = 0.0
        for j in range(n):
            sum += b_ij[(i, j)]
        model.addConstr(1 == sum, name=f"Population_constraint{i}")

    # Total unit constraint, sum(x_i) == m
    
    sum = 0.0
    for j in range(n):
        sum += z_j[j]
    model.addConstr(sum == m, name = f"unit_constraint")

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
                if b_ij[(i, j)]: ################## MODIFIED ####################
                    if b_ij[(i, j)].x != 0:
                        coms.append(i)
            if coms != []:
                assignments[j] = coms
                        
        ls = []
        for i in range(n):
            for j in range(n):
                if b_ij[(i, j)].x == 1 and j not in ls:
                    ls.append(j)
        """
        for j in range(n):
            vars = 0
            for i in range(n):
                vars += (b_ij[(i, j)].x) * p[i]
            if vars != 0:
                print(f"Unit on {j}'s total service is {vars} ")
        
        for i in range(n):
            for j in range(n):
                if abs((b_ij[(i, j)].x) * d_ij[i][j] * p[i] - Z.x) <= 0.01:
                    print(f"Z is unit on {j} sending service to {i}")
        """

        return str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x), ls, assignments
    else:
        return "Model Is Infeasible"
"""
GLOBAL OPTIMAL MODEL 2
 - Global optimal model with new Z constraint
"""
def sam_global2(dataset_index):

    # INITIAL DATA

    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    line1 = lines[0].split()
    n = int(line1[0]) # Number of
    m = int(line1[1]) # Number of healthcare units to place
    M = n

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

    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: 1 if population i is served by unit on j. 0 otherwise
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype = GRB.BINARY, name = f"b_{i}_{j}")

    # z_j: 1 if facility is opened at node j. 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype = GRB.BINARY, name = f"z_{j}"))

    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 
    
    model.update()

    model.setObjective(Z, GRB.MINIMIZE)
    
    # Constraints
    
    # Z > p[i] * b_ij * d_ij for every i, j
    for i in range(n):
        model.addConstr(Z >= quicksum(p[i] * b_ij[i, j] * d_ij[i][j] for j in range(n)), name = f"Z_constraint_{i}")

    # Capacity constraint, sum(b_ij * p[i]) <= C for every unit. (There can be surplus capacity ?)
    for j in range(n):
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, j)] * p[i]
        model.addConstr(variables <= C * z_j[j], name = f"capacity_constraint_{j}")
  
    # Population constraint
    for i in range(n):
        sum = 0.0
        for j in range(n):
            sum += b_ij[(i, j)]
        model.addConstr(1 == sum, name=f"Population_constraint{i}")

    # Total unit constraint, sum(x_i) == m
    
    sum = 0.0
    for j in range(n):
        sum += z_j[j]
    model.addConstr(sum == m, name = f"unit_constraint")

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
                if b_ij[(i, j)]: ################## MODIFIED ####################
                    if b_ij[(i, j)].x != 0:
                        coms.append(i)
            if coms != []:
                assignments[j] = coms
                        
        ls = []
        for i in range(n):
            for j in range(n):
                if b_ij[(i, j)].x == 1 and j not in ls:
                    ls.append(j)
        """
        for j in range(n):
            vars = 0
            for i in range(n):
                vars += (b_ij[(i, j)].x) * p[i]
            if vars != 0:
                print(f"Unit on {j}'s total service is {vars} ")
        
        for i in range(n):
            for j in range(n):
                if abs((b_ij[(i, j)].x) * d_ij[i][j] * p[i] - Z.x) <= 0.01:
                    print(f"Z is unit on {j} sending service to {i}")
        """

        return str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x), ls, assignments
    else:
        return "Model Is Infeasible"
"""
FIRST METHOD OF APPROXIMATION
 - If two nodes population weighted distance is too large, it is unlikely that if a unit is placed on one of them, it will serve the other
 - If the population weighted distance between node i and j d_ij >= d_max, don't create variable b_ij
"""
def sam_apx1(dataset_index, d_up = 90):

    # INITIAL DATA

    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    line1 = lines[0].split()
    n = int(line1[0]) # Number of
    m = int(line1[1]) # Number of healthcare units to place
    M = n

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

    all_weighted_distances = [d_ij[i][j] * ((p[i] + p[j]) / 2) for i in range(n) for j in range(n) if i != j] ################## MODIFIED ####################
    d_max = percentile(all_weighted_distances, d_up) ################## MODIFIED ####################

    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: if population i is served by unit on j
    b_ij = {}
    for i in range(n):
        for j in range(n):
            if d_ij[i][j] < d_max: ################## MODIFIED ####################
                b_ij[(i, j)] = model.addVar(vtype = GRB.BINARY, name = f"b_{i}_{j}")
            else:
                b_ij[(i, j)] = None

    # z_j: 1 if facility is opened at node j. 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype = GRB.BINARY, name = f"z_{j}"))

    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 
    
    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    
    # Z > b_ij * d_ij for every i, j
    for i in range(n):
        for j in range(n):
            if i != j and b_ij[(i, j)]: ################## MODIFIED ####################
                model.addConstr(Z >= p[i] * b_ij[i, j] * d_ij[i][j], name = f"Z_constraint_{i}_{j}")
    
    # Capacity constraint, sum(b_ij * p[i]) <= C for every unit. (There can be surplus capacity ?)
    for j in range(n):
        variables = 0.0
        for i in range(n):
            if b_ij[(i, j)]:
                variables += b_ij[(i, j)] * p[i]
        model.addConstr(variables <= C * z_j[j], name = f"capacity_constraint_{i}")
  
    # Population constraint
    for i in range(n):
        sum = 0.0
        for j in range(n):
            if b_ij[(i, j)]: ################## MODIFIED ####################
                sum += b_ij[(i, j)]
        model.addConstr(1 == sum, name=f"Population_constraint{i}")

    # Total unit constraint, sum(x_i) == m

    sum = 0.0
    for j in range(n):
        sum += z_j[j]
    model.addConstr(sum == m, name = f"unit_constraint")

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
                if b_ij[(i, j)]: ################## MODIFIED ####################
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
        return str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x), ls, assignments
    else:
        return "Model Is Infeasible"
"""
SECOND METHOD OF APPROXIMATION
 - In addition to methods in apx1
 - If p[i] is small, Z >= d_ij * b_ij * p[i] doesn't need to be enforced. 
"""
def sam_apx2(dataset_index, d_up = 90, p_lp = 10):

    # INITIAL DATA

    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    line1 = lines[0].split()
    n = int(line1[0]) # Number of
    m = int(line1[1]) # Number of healthcare units to place
    M = n

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

    all_distances = [d_ij[i][j] * (p[i] + p[j])/2 for i in range(n) for j in range(n) if i != j] ################## MODIFIED ####################
    d_max = percentile(all_distances, d_up) ################## MODIFIED ####################

    p_min = percentile(p, p_lp)
    print(f"Lowest population allowed: {p_min}")
    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: if population i is served by unit on j
    b_ij = {}
    for i in range(n):
        for j in range(n):
            if d_ij[i][j] < d_max: ################## MODIFIED ####################
                b_ij[(i, j)] = model.addVar(vtype = GRB.BINARY, name = f"b_{i}_{j}")
            else:
                b_ij[(i, j)] = None

    # z_j: 1 if facility is opened at node j. 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype = GRB.BINARY, name = f"z_{j}"))

    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 
    
    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    
    # Z > b_ij * d_ij * p[i] for every i, j 
    # Only add if p[i] isn't very small 
    for i in range(n):
        for j in range(n):
            if i != j and b_ij[(i, j)] and p[i] >= p_min: ################## MODIFIED ####################
                model.addConstr(Z >= p[i] * b_ij[i, j] * d_ij[i][j], name = f"Z_constraint_{i}_{j}")
    
    # Capacity constraint, sum(b_ij * p[i]) <= C for every unit. (There can be surplus capacity ?)
    for j in range(n):
        variables = 0.0
        for i in range(n):
            if b_ij[(i, j)]:
                variables += b_ij[(i, j)] * p[i]
        model.addConstr(variables <= C * z_j[j], name = f"capacity_constraint_{i}")
  
    # Population constraint
    for i in range(n):
        sum = 0.0
        for j in range(n):
            if b_ij[(i, j)]: ################## MODIFIED ####################
                sum += b_ij[(i, j)]
        model.addConstr(1 == sum, name=f"Population_constraint{i}")

    # Total unit constraint, sum(x_i) == m

    sum = 0.0
    for j in range(n):
        sum += z_j[j]
    model.addConstr(sum == m, name = f"unit_constraint")

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
                if b_ij[(i, j)]: ################## MODIFIED ####################
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
        return str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x), ls, assignments
    else:
        return "Model Is Infeasible"
"""
INITIAL FEASIBLE SOLUTION
 - Generation of ifs with assignment problem
 - Pre-located units by agglomerative clustering, solve it as assignment problem, use the solution as an ifs
"""
def sam_ifs1(dataset_index, units: list):

    # INITIAL DATA

    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    line1 = lines[0].split()
    n = int(line1[0]) # Number of
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

    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    b_ij = {} # b_ij = 1 if community i is served by unit on j
    for unit_index in units:
        for i in range(n):
            b_ij[(i, unit_index)] = model.addVar(vtype = GRB.BINARY, name = f"b_{i}_{unit_index}")  

    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 
    
    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    # Z constraint
    for unit_index in units:
        for i in range(n):
            if i != unit_index:
                model.addConstr(Z >= b_ij[(i, unit_index)] * d_ij[i][unit_index] * p[i], name = f"Z_constraint_{i}_{unit_index}")

    # Single allocation constraint, sum(b_ij) == 1 for every population
    for i in range(n):
        variables = 0.0
        for unit_index in units:
            variables += b_ij[(i, unit_index)]
        model.addConstr(variables == 1, name = f"single_allocation_constraint_{i}")

    # Capacity constraint
    for unit_index in units:
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, unit_index)] * p[i]
        model.addConstr(variables <= C, name = f"capacity_constraint_{unit_index}")
    
    
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

        return Z.x
    else:
        return "Model Is Infeasible"
"""
GLOBAL SOLUTION WITH IFS1
 - Using sam_ifs1() to generate an ifs
 - Using the ifs to solve for global optimum
"""
def sam_global_w_ifs1(dataset_index):

    # INITIAL DATA

    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    line1 = lines[0].split()
    n = int(line1[0]) # Number of
    m = int(line1[1]) # Number of healthcare units to place
    M = n

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

    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: if population i is served by unit on j
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype = GRB.BINARY, name = f"b_{i}_{j}")

    # z_j: 1 if facility is opened at node j. 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype = GRB.BINARY, name = f"z_{j}"))

    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 
    
    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    
    # Z > b_ij * d_ij for every i, j
    for i in range(n):
        for j in range(n):
            model.addConstr(Z >= p[i] * b_ij[i, j] * d_ij[i][j], name = f"Z_constraint_{i}_{j}")
    
    # Capacity constraint, sum(b_ij * p[i]) <= C for every unit. (There can be surplus capacity ?)
    for j in range(n):
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, j)] * p[i]
        model.addConstr(variables <= C, name = f"capacity_constraint_{i}")
  
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
    model.addConstr(sum == m, name = f"unit_constraint")

    # START TIMER
    start = time()


    units = d_cluster(dataset_index)[0]
    b_ij_start, ifs_time = sam_ifs1(dataset_index, units)

    # Set starting values with IFS
    for i in range(n):
        for j in range(n):
            if (i, j) in b_ij_start:
                b_ij[(i, j)].start = b_ij_start[(i, j)]
            else:
                b_ij[(i, j)].start = 0
    

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
                if b_ij[(i, j)]: ################## MODIFIED ####################
                    if b_ij[(i, j)].x != 0:
                        coms.append(i)
            if coms != []:
                assignments[j] = coms
                        
        ls = []
        for i in range(n):
            for j in range(n):
                if b_ij[(i, j)].x == 1 and j not in ls:
                    ls.append(j)
        
        return f"IFS took {ifs_time} seconds. Total time: "+ str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x), ls, assignments
    else:
        return "Model Is Infeasible"
"""
GLOBAL SOLUTION 2 WITH IFS1
 - Using sam_ifs1() to generate an ifs
 - Global optimal model with new Z constraint
"""
def sam_global2_w_ifs1(dataset_index):

    # INITIAL DATA

    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    line1 = lines[0].split()
    n = int(line1[0]) # Number of
    m = int(line1[1]) # Number of healthcare units to place
    M = n

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

    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: if population i is served by unit on j
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype = GRB.BINARY, name = f"b_{i}_{j}")

    # z_j: 1 if facility is opened at node j. 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype = GRB.BINARY, name = f"z_{j}"))

    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 
    
    model.update()
    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    
    # Z > p[i] * b_ij * d_ij for every i, j
    for i in range(n):
        model.addConstr(Z >= quicksum(p[i] * b_ij[i, j] * d_ij[i][j] for j in range(n)), name = f"Z_constraint_{i}")
    
    # Capacity constraint, sum(b_ij * p[i]) <= C for every unit. (There can be surplus capacity ?)
    for j in range(n):
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, j)] * p[i]
        model.addConstr(variables <= C * z_j[j], name = f"capacity_constraint_{i}")
  
    # Population constraint
    for i in range(n):
        sum = 0.0
        for j in range(n):
            sum += b_ij[(i, j)]
        model.addConstr(1 == sum, name=f"Population_constraint{i}")

    sum = 0.0
    for j in range(n):
        sum += z_j[j]
    model.addConstr(sum == m, name = f"unit_constraint")

    # START TIMER
    start = time()


    units = d_cluster(dataset_index)[0]
    b_ij_start, ifs_time = sam_ifs1(dataset_index, units)

    # Set starting values with IFS
    for i in range(n):
        for j in range(n):
            if (i, j) in b_ij_start:
                b_ij[(i, j)].start = b_ij_start[(i, j)]
            else:
                b_ij[(i, j)].start = 0
    

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
                if b_ij[(i, j)]: ################## MODIFIED ####################
                    if b_ij[(i, j)].x != 0:
                        coms.append(i)
            if coms != []:
                assignments[j] = coms
                        
        ls = []
        for i in range(n):
            for j in range(n):
                if b_ij[(i, j)].x == 1 and j not in ls:
                    ls.append(j)
        
        return f"IFS took {ifs_time} seconds. Total time: "+ str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x), ls, assignments
    else:
        return "Model Is Infeasible"

def sam_new_obj(dataset_index):

    # INITIAL DATA

    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    line1 = lines[0].split()
    n = int(line1[0]) # Number of
    m = int(line1[1]) # Number of healthcare units to place
    M = n

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

    # Create model

    model = Model("Healthcare Placement")
    model.setParam("MIPGap", 0.01)
    # Decision variables

    # b_ij: 1 if population i is served by unit on j. 0 otherwise
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype = GRB.BINARY, name = f"b_{i}_{j}")

    # z_j: 1 if facility is opened at node j. 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype = GRB.BINARY, name = f"z_{j}"))

    model.update()

    Z = 0
    for i in range(n):
        for j in range(n):
            Z += b_ij[(i, j)] * p[i] * d_ij[i][j]
    
    model.setObjective(Z, GRB.MINIMIZE)
    
    # Constraints
    
    # Capacity constraint, sum(b_ij * p[i]) <= C for every unit. (There can be surplus capacity ?)
    for j in range(n):
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, j)] * p[i]
        model.addConstr(variables <= C * z_j[j], name = f"capacity_constraint_{j}")
  
    # Population constraint
    for i in range(n):
        sum = 0.0
        for j in range(n):
            sum += b_ij[(i, j)]
        model.addConstr(1 == sum, name=f"Population_constraint{i}")

    # Total unit constraint, sum(x_i) == m
    sum = 0.0
    for j in range(n):
        sum += z_j[j]
    model.addConstr(sum == m, name = f"unit_constraint")

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
                if b_ij[(i, j)]: ################## MODIFIED ####################
                    if b_ij[(i, j)].x != 0:
                        coms.append(i)
            if coms != []:
                assignments[j] = coms
                        
        ls = []
        for i in range(n):
            for j in range(n):
                if b_ij[(i, j)].x == 1 and j not in ls:
                    ls.append(j)
        
        als = []
        for i in range(n):
            for j in range(n):
                als.append((b_ij[(i, j)].x) * d_ij[i][j] * p[i])

        

        return str(round(time_elapsed, 2)) + " seconds, Z : " + str(max(als)), ls, assignments
    else:
        return "Model Is Infeasible"
"""
THIRD APPROXIMATION METHOD WITH IFS1
 - New Z constraint
 - Use of ifs1
 - Method of approximation 1 and 2
"""

def sam_apx3_w_ifs1(dataset_index, d_up = 90, p_lp = 10):

    # INITIAL DATA

    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    line1 = lines[0].split()
    n = int(line1[0]) # Number of
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

    all_weighted_distances = [d_ij[i][j] * ((p[i] + p[j])/2) for i in range(n) for j in range(n) if i != j] ################## MODIFIED ####################
    d_max = percentile(all_weighted_distances, d_up) ################## MODIFIED ####################
    p_min = percentile(p, p_lp) ################## MODIFIED ####################

    # Create model

    model = Model("Healthcare Placement")
    model.setParam("Threads", 11)
    # Decision variables

    # b_ij: if population i is served by unit on j
    b_ij = {}
    h = 0
    for i in range(n):
        for j in range(n):
            if d_ij[i][j] * ((p[i] + p[j])/2) < d_max: ################## MODIFIED ####################
                b_ij[(i, j)] = model.addVar(vtype = GRB.BINARY, name = f"b_{i}_{j}")
            else:
                b_ij[(i, j)] = None

    # z_j: 1 if facility is opened at node j. 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype = GRB.BINARY, name = f"z_{j}"))

    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 
    
    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    
    # Z > b_ij * d_ij * p[i] for every i, j 
    # Only add if p[i] isn't very small 
    for i in range(n):
        variables = 0.0
        for j in range(n):
            if i != j and b_ij[(i, j)] and p[i] >= p_min: ################## MODIFIED ####################
                variables += p[i] * b_ij[i, j] * d_ij[i][j]
        model.addConstr(Z >= variables, name = f"Z_constraint_{j}")
    
    # Capacity constraint, sum(b_ij * p[i]) <= C for every unit. (There can be surplus capacity ?)
    for j in range(n):
        variables = 0.0
        for i in range(n):
            if b_ij[(i, j)]:
                variables += b_ij[(i, j)] * p[i]
        model.addConstr(variables <= C * z_j[j], name = f"capacity_constraint_{i}")
  
    # Population constraint
    for i in range(n):
        sum = 0.0
        for j in range(n):
            if b_ij[(i, j)]: ################## MODIFIED ####################
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
            if b_ij[(i, j)]: ################## MODIFIED ####################
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
    return f"Optimality Gap: {model.MIPGap * 100:.2f}% " + str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x), ls, assignments

def csam_ifs1(dataset_index, units : list):
    # INITIAL DATA

    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    line1 = lines[0].split()
    n = int(line1[0]) # Number of
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

    alpha = round((sum(p) / m ) * 0.2)
    beta = max([d_ij[i][j] for i in range(n) for j in range(n)]) * 0.2

    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    b_ij = {} # b_ij = 1 if community i is served by unit on j
    for unit_index in units:
        for i in range(n):
            b_ij[(i, unit_index)] = model.addVar(vtype = GRB.BINARY, name = f"b_{i}_{unit_index}")

    # w_j: workload of unit j
    w_j = [model.addVar(vtype=GRB.CONTINUOUS, name=f"w_{j}") for j in range(n)]
    for unit_index in units:
        model.addConstr(w_j[unit_index] == quicksum(b_ij[(i, unit_index)] * p[i] for i in range(n)))

    # Distance each community travels to get service
    d_i = [model.addVar(vtype=GRB.CONTINUOUS, name=f"d_{i}") for i in range(n)]
    for i in range(n):
        model.addConstr(d_i[i] == quicksum(b_ij[(i, unit_index)] * d_ij[i][unit_index] for unit_index in units), name=f"d_{i}")

    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 
    

    model.update()
    
    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints
    
    # Alpha constraint
    for i in units:
        for k in units:
            if i != k:
                model.addConstr(w_j[i] - w_j[k] <= alpha, name = f"alpha_constraint_{i}{k}")
                model.addConstr(w_j[k] - w_j[i] <= alpha, name = f"alpha_constraint_{k}{i}")
    
    # Beta constraint
    for i in range(n):
        for j in range(n):
            if i != j:
                model.addConstr(d_i[i] - d_i[j] <= beta, name = f"beta_constraint_{i}{j}")
                model.addConstr(d_i[j] - d_i[i] <= beta, name = f"beta_constraint_{j}{i}")
    
     # Z constraint
    for unit_index in units:
        for i in range(n):
            if i != unit_index:
                model.addConstr(Z >= b_ij[(i, unit_index)] * d_ij[i][unit_index] * p[i], name = f"Z_constraint_{i}_{unit_index}")

    # Single allocation constraint, sum(b_ij) == 1 for every population
    for i in range(n):
        variables = 0.0
        for unit_index in units:
            variables += b_ij[(i, unit_index)]
        model.addConstr(variables == 1, name = f"single_allocation_constraint_{i}")

   # Capacity constraint
    for unit_index in units:
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, unit_index)] * p[i]
        model.addConstr(variables <= C, name = f"capacity_constraint_{unit_index}")
    
    
    model.update()

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

        assignments = {}
        for j in units:
            coms = []
            for i in range(n):
                if b_ij[(i, j)] != 0:
                    coms.append(i)
            if coms != []:
                assignments[j] = coms
        w = []
        for i in units:
            if w_j[i].x != 0:
                w.append(w_j[i].x)
        d = []
        for i in range(n):
            d.append(d_i[i].x)
        
        return b_ij, round(time_elapsed, 2)
    else:
        return "Infeasible"
    
def csam_global(dataset_index):

    # INITIAL DATA
    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    line1 = lines[0].split()
    n = int(line1[0]) # Number of
    m = int(line1[1]) # Number of healthcare units to place

    for line in lines[2:]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, C, population = values[1], values[2], values[3], int(values[4])
        cord_com.append([x, y]) # add coordinates
        p.append(population)  # add populations for each community i



    # Calculate distances
    d_ij = {}
    i = 0
    for x1, y1 in cord_com:
        j = 0
        for x2, y2 in cord_com:
            distance = dist((x1, y1), (x2, y2))
            d_ij[(i, j)] = distance
            j += 1
        i += 1


    alpha = round((sum(p) / m ) * 0.2)
    beta = max([d_ij[(i, j)] for i in range(n) for j in range(n)]) * 0.2
    M1 = 10000

    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: 1 if population i is served by unit on j. 0 otherwise
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype = GRB.BINARY, name = f"b_{i}_{j}")

    # z_j: 1 if facility is opened at node j. 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype = GRB.BINARY, name = f"z_{j}")) 

    # w_j: workload of node j
    w_j = [model.addVar(vtype=GRB.CONTINUOUS, name=f"w_{j}") for j in range(n)]
    for j in range(n):
        model.addConstr(w_j[j] == quicksum(p[i] * b_ij[i, j] for i in range(n)), name=f"workload_def_{j}")
    
    # Distance each community travels to get service
    d_i = [model.addVar(vtype=GRB.CONTINUOUS, name=f"d_{i}") for i in range(n)]
    for i in range(n):
        model.addConstr(d_i[i] == quicksum(b_ij[(i, j)] * d_ij[(i, j)] for j in range(n)), name=f"d_{i}")
    
    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 
    
    model.update()
    model.setParam("Threads", 11)  # Use at most 4 threads

    model.setObjective(Z, GRB.MINIMIZE)
    
    # Constraints

    # Z > p[i] * b_ij * d_ij for every i, j
    for i in range(n):
        model.addConstr(Z >= p[i] * d_i[i], name = f"Z_constraint_{j}")
    
    # Beta fairness constraint
    for i in range(n):
        for k in range(i):
            #diff = model.addVar(vtype = GRB.CONTINUOUS)
            #model.addConstr(diff >= d_i[i] - d_i[k], name=f"wasd_pos_{i}_{k}")
            #model.addConstr(diff >= - d_i[i] + d_i[k], name=f"w_asdneg_{i}_{k}")
            #model.addConstr(diff <= beta)

            model.addConstr(beta >= d_i[i] - d_i[k], name=f"beta_diff_pos_{i}_{k}")
            model.addConstr(beta >= d_i[k] - d_i[i], name=f"beta_diff_neg_{i}_{k}")
            
    # Alpha fairness constraint
    for i in range(n):
        for j in range(i):
            if i != j:
                #diff = model.addVar(vtype = GRB.CONTINUOUS)
                #model.addConstr(diff >= w_j[i] - w_j[j], name=f"w_pos_{i}_{j}")
                #model.addConstr(diff >= - w_j[i] + w_j[j], name=f"w_neg_{i}_{j}")
                #model.addConstr(diff <= alpha + M1 * (2 - z_j[i] - z_j[j]))

                model.addConstr(alpha + M1 * (2 - z_j[i] - z_j[j]) >= w_j[i] - w_j[j], name=f"w_pos_{i}_{j}")
                model.addConstr(alpha + M1 * (2 - z_j[i] - z_j[j]) >= w_j[j] - w_j[i], name=f"w_neg_{i}_{j}")
            
    # Capacity constraint, sum(b_ij * p[i]) <= C for every unit. (There can be surplus capacity ?)
    for j in range(n):
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, j)] * p[i]
        model.addConstr(variables <= C * z_j[j], name = f"capacity_constraint_{j}")
  
    # Population constraint
    for i in range(n):
        var = 0.0
        for j in range(n):
            var += b_ij[(i, j)]
        model.addConstr(1 == var, name=f"Population_constraint{i}")

    # Total unit constraint, sum(x_i) <= m
    var = 0.0
    for j in range(n):
        var += z_j[j]
    model.addConstr(var <= m, name = f"unit_constraint")

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
                if b_ij[(i, j)]: ################## MODIFIED ####################
                    if b_ij[(i, j)].x != 0:
                        coms.append(i)
            if coms != []:
                assignments[j] = coms
        w = []
        for i in range(n):
            if w_j[i].x != 0:
                w.append(w_j[i].x)
        d = []
        for i in range(len(d_i)):
            d.append(d_i[i].x)      
        ls = []
        for i in range(n):
            for j in range(n):
                if b_ij[(i, j)].x == 1 and j not in ls:
                    ls.append(j)
        return str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x), ls, max(w), min(w), alpha, max(d), min(d), beta, assignments
    else:
        return "Model Is Infeasible"

def csam_global_w_ifs1(dataset_index):
    # INITIAL DATA
    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    line1 = lines[0].split()
    n = int(line1[0]) # Number of
    m = int(line1[1]) # Number of healthcare units to place

    for line in lines[2:]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, C, population = values[1], values[2], values[3], int(values[4])
        cord_com.append([x, y]) # add coordinates
        p.append(population)  # add populations for each community i



    # Calculate distances
    d_ij = {}
    i = 0
    for x1, y1 in cord_com:
        j = 0
        for x2, y2 in cord_com:
            distance = dist((x1, y1), (x2, y2))
            d_ij[(i, j)] = distance
            j += 1
        i += 1

    alpha = round((sum(p) / m ) * 0.2)
    beta = max([d_ij[(i, j)] for i in range(n) for j in range(n)]) * 0.2
    M1 = 10000

    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: 1 if population i is served by unit on j. 0 otherwise
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype = GRB.BINARY, name = f"b_{i}_{j}")

    # z_j: 1 if facility is opened at node j. 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype = GRB.BINARY, name = f"z_{j}")) 

    # w_j: workload of node j
    w_j = [model.addVar(vtype=GRB.CONTINUOUS, name=f"w_{j}") for j in range(n)]
    for j in range(n):
        model.addConstr(w_j[j] == quicksum(p[i] * b_ij[i, j] for i in range(n)), name=f"workload_def_{j}")
    
    # Distance each community travels to get service
    d_i = [model.addVar(vtype=GRB.CONTINUOUS, name=f"d_{i}") for i in range(n)]
    for i in range(n):
        model.addConstr(d_i[i] == quicksum(b_ij[(i, j)] * d_ij[(i, j)] for j in range(n)), name=f"d_{i}")
    
    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 
    
    model.update()

    model.setObjective(Z, GRB.MINIMIZE)
    model.setParam("Threads", 10)
    # Constraints

    # Z > p[i] * b_ij * d_ij for every i, j
    for i in range(n):
        model.addConstr(Z >= p[i] * d_i[i], name = f"Z_constraint_{j}")
    
    # Beta fairness constraint
    for i in range(n):
        for k in range(i):
            model.addConstr(beta >= d_i[i] - d_i[k], name=f"beta_diff_pos_{i}_{k}")
            model.addConstr(beta >= d_i[k] - d_i[i], name=f"beta_diff_neg_{i}_{k}")
            
    # Alpha fairness constraint
    for i in range(n):
        for j in range(i):
            if i != j:
                model.addConstr(alpha + M1 * (2 - z_j[i] - z_j[j]) >= w_j[i] - w_j[j], name=f"w_pos_{i}_{j}")
                model.addConstr(alpha + M1 * (2 - z_j[i] - z_j[j]) >= w_j[j] - w_j[i], name=f"w_neg_{i}_{j}")
            
    # Capacity constraint, sum(b_ij * p[i]) <= C for every unit. (There can be surplus capacity ?)
    for j in range(n):
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, j)] * p[i]
        model.addConstr(variables <= C * z_j[j], name = f"capacity_constraint_{j}")
  
    # Population constraint
    for i in range(n):
        var = 0.0
        for j in range(n):
            var += b_ij[(i, j)]
        model.addConstr(1 == var, name=f"Population_constraint{i}")

    # Total unit constraint, sum(x_i) <= m
    var = 0.0
    for j in range(n):
        var += z_j[j]
    model.addConstr(var <= m, name = f"unit_constraint")

    # START TIMER
    start = time()

    units = d_cluster_new(dataset_index)[0]
    b_ij_start, ifs_time = csam_ifs1(dataset_index, units)

    # Set starting values of b_ij's with IFS
    for i in range(n):
        for j in range(n):
            if (i, j) in b_ij_start:
                b_ij[(i, j)].start = b_ij_start[(i, j)]
            else:
                b_ij[(i, j)].start = 0
    # Set starting values of z_j's with IFS
    for j in range(n):
        if j in units:
            z_j[j].start = 1
        else:
            z_j[j].start = 0
    
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
            if b_ij[(i, j)]: ################## MODIFIED ####################
                if b_ij[(i, j)].x != 0:
                    coms.append(i)
        if coms != []:
            assignments[j] = coms
    w = []
    for i in range(n):
        if w_j[i].x != 0:
            w.append(w_j[i].x)
    d = []
    for i in range(len(d_i)):
        d.append(d_i[i].x)      
    ls = []
    for i in range(n):
        for j in range(n):
            if b_ij[(i, j)].x == 1 and j not in ls:
                ls.append(j)
    return str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x), ls, max(w), min(w), alpha, max(d), min(d), beta, assignments

def csam_apx1_w_ifs1(dataset_index, d_up):
    
    # INITIAL DATA
    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    line1 = lines[0].split()
    n = int(line1[0]) # Number of
    m = int(line1[1]) # Number of healthcare units to place

    for line in lines[2:]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, C, population = values[1], values[2], values[3], int(values[4])
        cord_com.append([x, y]) # add coordinates
        p.append(population)  # add populations for each community i



    # Calculate distances
    d_ij = {}
    i = 0
    for x1, y1 in cord_com:
        j = 0
        for x2, y2 in cord_com:
            distance = dist((x1, y1), (x2, y2))
            d_ij[(i, j)] = distance
            j += 1
        i += 1

    all_weighted_distances = [d_ij[(i, j)] * ((p[i] + p[j]) / 2) for i in range(n) for j in range(n) if i != j]
    d_max = percentile(all_weighted_distances, d_up)
    alpha = round((sum(p) / m ) * 0.2)
    beta = max([d_ij[(i, j)] for i in range(n) for j in range(n)]) * 0.2
    M1 = 10000

    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: 1 if population i is served by unit on j. 0 otherwise
    b_ij = {}
    for i in range(n):
        for j in range(n):
            if d_ij[(i, j)] <= d_max:
                b_ij[(i, j)] = model.addVar(vtype = GRB.BINARY, name = f"b_{i}_{j}")

    # z_j: 1 if facility is opened at node j. 0 o/w
    z_j = []
    for j in range(n):
        z_j.append(model.addVar(vtype = GRB.BINARY, name = f"z_{j}")) 

    # w_j: workload of node j
    w_j = [model.addVar(vtype=GRB.CONTINUOUS, name=f"w_{j}") for j in range(n)]
    for j in range(n):
        model.addConstr(w_j[j] == quicksum(p[i] * b_ij[i, j] for i in range(n)), name=f"workload_def_{j}")
    
    # Distance each community travels to get service
    d_i = [model.addVar(vtype=GRB.CONTINUOUS, name=f"d_{i}") for i in range(n)]
    for i in range(n):
        model.addConstr(d_i[i] == quicksum(b_ij[(i, j)] * d_ij[(i, j)] for j in range(n) if (i, j) in b_ij.keys()), name=f"d_{i}")
    
    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 
    
    model.update()

    model.setObjective(Z, GRB.MINIMIZE)
    
    # Constraints

    # Z > p[i] * b_ij * d_ij for every i, j
    for i in range(n):
        model.addConstr(Z >= p[i] * d_i[i], name = f"Z_constraint_{j}")
    
    # Beta fairness constraint
    for i in range(n):
        for k in range(i):
            model.addConstr(beta >= d_i[i] - d_i[k], name=f"beta_diff_pos_{i}_{k}")
            model.addConstr(beta >= d_i[k] - d_i[i], name=f"beta_diff_neg_{i}_{k}")
            
    # Alpha fairness constraint
    for i in range(n):
        for j in range(i):
            if i != j:
                model.addConstr(alpha + M1 * (2 - z_j[i] - z_j[j]) >= w_j[i] - w_j[j], name=f"w_pos_{i}_{j}")
                model.addConstr(alpha + M1 * (2 - z_j[i] - z_j[j]) >= w_j[j] - w_j[i], name=f"w_neg_{i}_{j}")
            
    # Capacity constraint, sum(b_ij * p[i]) <= C for every unit. (There can be surplus capacity ?)
    for j in range(n):
        variables = 0.0
        for i in range(n):
            if (i, j) in b_ij.keys():
                variables += b_ij[(i, j)] * p[i]
        model.addConstr(variables <= C * z_j[j], name = f"capacity_constraint_{j}")
  
    # Population constraint
    for i in range(n):
        var = 0.0
        for j in range(n):
            if (i, j) in b_ij.keys():
                var += b_ij[(i, j)]
        model.addConstr(1 == var, name=f"Population_constraint{i}")

    # Total unit constraint, sum(x_i) <= m
    var = 0.0
    for j in range(n):
        var += z_j[j]
    model.addConstr(var <= m, name = f"unit_constraint")

    # START TIMER
    start = time()

    units = d_cluster_new(dataset_index)[0]
    b_ij_start, ifs_time = csam_ifs1(dataset_index, units)

    # Set starting values of b_ij's with IFS
    for i in range(n):
        for j in range(n):
            if (i, j) in b_ij.keys():
                if (i, j) in b_ij_start:
                    b_ij[(i, j)].start = b_ij_start[(i, j)]
                else:
                    b_ij[(i, j)].start = 0
    # Set starting values of z_j's with IFS
    for j in range(n):
        if j in units:
            z_j[j].start = 1
        else:
            z_j[j].start = 0
    
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
                if (i, j) in b_ij.keys(): ################## MODIFIED ####################
                    if b_ij[(i, j)].x != 0:
                        coms.append(i)
            if coms != []:
                assignments[j] = coms
        w = []
        for i in range(n):
            if w_j[i].x != 0:
                w.append(w_j[i].x)
        d = []
        for i in range(len(d_i)):
            d.append(d_i[i].x)      
        ls = []
        for i in range(n):
            for j in range(n):
                if b_ij[(i, j)].x == 1 and j not in ls:
                    ls.append(j)
        return str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x), ls, max(w), min(w), alpha, max(d), min(d), beta, assignments
    else:
        return "Model Is Infeasible"
    
print(csam_global_w_ifs1(11))