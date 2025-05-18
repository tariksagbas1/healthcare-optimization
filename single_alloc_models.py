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

    # Presolve constraints
    for j in range(n):
        for i in range(n):
            model.addConstr(b_ij[i,j] <= z_j[j])

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
    M1 = C

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
    model.setParam("MIPFocus", 1)
    model.setParam("Heuristics", 0.5)
    model.setParam("Cuts", 2)

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

    for j in range(n):
        for i in range(n):
            model.addConstr(b_ij[i,j] <= z_j[j])

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
    M1 = C

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
    
def cifcif(dataset_index):
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
    M1 = C

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
    model.setParam("MIPFocus", 1)
    model.setParam("Heuristics", 0.5)
    model.setParam("Cuts", 2)

    # Constraints

    # Z > p[i] * b_ij * d_ij for every i, j
    for i in range(n):
        model.addConstr(Z >= p[i] * d_i[i], name = f"Z_constraint_{j}")
    
    # Beta fairness constraint
    for i in range(n):
        model.addConstr(beta >= d_i[i], name=f"beta_diff_pos_{i}")
            
    # Alpha fairness constraint
    for i in range(n):
        model.addConstr(alpha + M1 * (1 - z_j[i]) >= C - w_j[i], name=f"w_pos_{i}_{j}")
            
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

    for j in range(n):
        for i in range(n):
            model.addConstr(b_ij[i,j] <= z_j[j])

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

def csam_global_w_assignments(dataset_index, uns : list, assignments : dict):
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
    M1 = C

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
    model.setParam("MIPFocus", 1)
    model.setParam("Heuristics", 0.5)

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

    for j in range(n):
        for i in range(n):
            model.addConstr(b_ij[i,j] <= z_j[j])

    # Assign starting values for b_ij
    for j, comms in assignments.items():
        for i in comms:
            b_ij[(i, j)].start = 1
    
    # Assign starting values for z_j
    for j in range(n):
        if j in uns:
            z_j[j].start = 1
        else:
            z_j[j].start = 0
        
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
    model.setParam("Presolve", 1)
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

    # Presolve constraints
    for j in range(n):
        for i in range(n):
            model.addConstr(b_ij[i,j] <= z_j[j])

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
print(csam_global_w_assignments(16, [176, 1, 261, 128, 112, 82, 165, 264, 302, 9, 419, 11, 225, 200, 499, 279, 179, 284, 342, 136, 381, 370, 25, 451, 44, 481, 57, 360, 72, 428, 412, 240, 364, 332, 163, 48, 266, 51, 64, 312, 305, 315, 338, 154, 494, 66, 81, 32, 265, 483, 463, 371, 108, 445, 208, 263, 91, 355, 363, 138, 492, 239, 392, 212, 437, 420, 365, 246, 441, 17, 122, 121, 443, 488, 130, 334, 271, 146, 196, 434, 386, 447, 189, 142, 318, 337, 241, 242, 248, 71], {1: [1, 41, 143, 202, 484], 9: [9, 116, 129, 191, 232, 330], 11: [11, 43, 80, 325, 451, 488, 491], 17: [121, 249, 411, 419], 25: [25, 164, 396, 450, 493], 32: [68, 155, 171, 262, 298, 331, 498], 44: [27, 376, 383, 453, 486], 48: [48, 50, 295, 436, 454], 51: [51, 234, 327, 333, 361], 57: [29, 57, 95, 113, 268, 308, 310, 480], 64: [55, 64, 161, 235, 496], 66: [66, 138, 339, 452], 71: [259, 271, 409, 415, 457], 72: [33, 37, 92, 300, 328], 81: [67, 81, 154, 206, 297, 432], 82: [5, 82, 309, 314, 467], 91: [85, 137, 165, 246], 108: [76, 182, 190, 420], 112: [4, 14, 79, 112, 245, 349, 416], 121: [124, 134, 140, 270, 426, 448], 122: [122, 178, 180, 186, 332, 378, 487], 128: [3, 12, 128, 169, 238, 329, 352, 476], 130: [130, 221, 313, 319, 455], 136: [22, 44, 63, 135, 267, 341, 348], 138: [93, 158, 274, 399, 410, 479], 142: [218, 335, 364, 387], 146: [146, 157, 216, 220, 285], 154: [61, 214, 370, 421, 499], 163: [47, 152, 163, 334], 165: [6, 91, 394, 406], 176: [0, 104, 109, 184, 350], 179: [18, 179, 197, 209, 424], 189: [189, 260, 311, 343, 423, 438, 461], 196: [147, 196, 439, 495], 200: [15, 200, 283, 401, 497], 208: [78, 166, 208, 275, 317, 321], 212: [101, 103, 183, 212, 224, 459], 225: [13, 19, 72, 324], 239: [98, 144, 195, 254, 287, 299, 345], 240: [40, 86, 89, 187, 240, 342], 241: [239, 241, 291, 292, 353, 464], 242: [242, 273, 305, 400, 468, 475], 246: [117, 176, 203, 247, 366], 248: [248, 354, 357, 393, 422, 428], 261: [2, 38, 173, 261, 263, 433], 263: [84, 170, 237, 255, 362], 264: [7, 230, 252, 278, 365, 407], 265: [69, 174, 207, 265, 301, 303, 358, 397, 465], 266: [49, 106, 160, 192, 251, 266, 282, 466], 271: [133, 210, 256, 318, 460], 279: [17, 131, 264, 279, 444], 284: [20, 244, 284, 385, 418], 302: [8, 52, 302, 316, 368], 305: [58, 153, 168, 185, 359, 377], 312: [56, 111, 312, 417], 315: [59, 97, 201, 429, 470], 318: [229, 257, 403, 425], 332: [46, 70, 177, 286, 294, 431, 442, 482], 334: [132, 175, 381, 427, 440], 337: [236, 258, 306, 389], 338: [60, 65, 99, 102, 194, 338], 342: [21, 367, 435, 458, 489], 355: [88, 148, 198, 205, 269, 276, 355], 360: [31, 35, 215, 223, 360, 477], 363: [90, 141, 289, 326, 363], 364: [42, 73, 142, 473], 365: [114, 150, 233, 307, 340], 370: [24, 53, 119, 430], 371: [75, 172, 228, 290, 371], 381: [23, 30, 54, 120, 199, 322, 469, 478], 386: [181, 280, 304, 315, 386, 449], 392: [100, 105, 225, 227, 336], 412: [39, 115, 204, 213, 412], 419: [10, 139, 156, 347, 375, 408, 462], 420: [108, 296, 323, 379, 446], 428: [36, 162, 380, 398, 485], 434: [149, 272, 382, 434], 437: [107, 288, 437], 441: [118, 374, 405, 441, 456], 443: [125, 136, 145, 356, 414, 443, 494], 445: [77, 217, 445, 471], 447: [188, 320, 346, 392], 451: [26, 45, 123, 151, 231, 369, 373, 447], 463: [74, 193, 384, 463], 481: [28, 32, 34, 243, 277, 391, 481], 483: [71, 159, 226, 250, 483], 488: [127, 167, 211, 219, 293, 351, 474], 492: [94, 96, 110, 126, 337, 344, 402, 413, 472, 490, 492], 494: [62, 87, 253, 372, 388, 395], 499: [16, 83, 222, 281, 390, 404]}))

