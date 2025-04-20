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

        return b_ij, round(time_elapsed, 2)
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
    model.setParam("Threads", 10)
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

def cvrp_global(dataset_index, units = list(), assignments = dict()):
    full_units = [0] + units
    unit_cords = [] # Healthcare unit coordinates
    q = {} # Unit equipment need
    p = [] # Population of every community
    
    M = 50
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file:
        lines = file.readlines()
    
    line2 = lines[1].split()
    depot_cord = (float(line2[1]), float(line2[2])) # Depot coordinates
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
                x_ijk[(i, j, k)] = model.addVar(vtype = GRB.BINARY, name = f"x_{i}{j}{k}")
    
    # Auxillary variable
    u_i = {}
    for i in units:
        u_i[i] = model.addVar(vtype = GRB.CONTINUOUS, lb = q[i], ub = 10000)

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
        model.addConstr(total == 1, name = f"single_vehicle_constraint_{i}_{k}")
    
    total = 0.0
    for k in range(M):
        for j in full_units:
            if 0 != j:
                total += x_ijk[(0, j, k)]
    model.addConstr(total <= M, name = f"single_vehicle_constraint_{i}_{k}")
    
    
    for k in range(M):
        total = 0
        for i in units:
            for j in units:
                if i != j:
                    total += x_ijk[(i, j, k)] * q[j]
        model.addConstr(total <= 10000 * y_k[k], name = f"equipment_constraint_{k}")
    
    # Must include starting 0 node
    for k in range(M):
        total = 0.0
        for j in units:
            total += x_ijk[(0, j, k)]
        model.addConstr(total == 1 * y_k[k], name = f"starting_node_constraint_{k}")
    
    # Must include ending 0 node
    for k in range(M):
        total = 0.0
        for i in units:
            total += x_ijk[(i, 0, k)]
        model.addConstr(total == 1 * y_k[k], name = f"ending_node_constraint_{k}")

    # No going back to the same node you came from
    total = 0.0
    for k in range(M):
        for i in full_units:
            total += x_ijk[(i, i, k)]
    model.addConstr(total == 0, name = "No_self_loops_constraint")

    
    # Flow constraint
    for i in units:
        for k in range(M):
            outflow = 0.0
            inflow = 0.0
            for j in full_units:
                if i != j:
                    outflow += x_ijk[(i, j, k)]
                    inflow += x_ijk[(j, i, k)]
            model.addConstr(outflow == inflow, name = f"flow_constraint_{i}_{k}")
    
    # Ambulance count constraint
    total = 0.0
    for k in range(M):
        total += y_k[k]
    model.addConstr( total <= len(units))
    
    # Subtour elimination constraint
    for i in units:
        for j in units:
            if i != j:
                for k in range(M):
                    model.addConstr(u_i[i] - u_i[j] + (10000 * x_ijk[(i, j, k)]) <= 10000 - q[j], name = f"SEC_{(i, j, k)}")
    


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

#a = [365, 1, 25, 182, 148, 570, 6, 742, 595, 922, 590, 451, 529, 13, 246, 616, 17, 632, 650, 705, 21, 191, 119, 832, 879, 40, 29, 431, 739, 208, 57, 608, 325, 36, 539, 819, 788, 3, 509, 115, 198, 527, 64, 640, 50, 968, 176, 544, 815, 478, 190, 63, 416, 218, 897, 728, 423, 611, 489, 88, 204, 75, 76, 77, 72, 99, 81, 844, 950, 286, 507, 504, 943, 399, 335, 482, 990, 220, 302, 126, 973, 113, 707, 425, 257, 336, 777, 179, 469, 837, 501, 123, 124, 125, 937, 206, 253, 694, 748, 628, 49, 891, 352, 402, 316, 525, 503, 164, 957, 229, 870, 330, 848, 622, 594, 323, 193, 345, 291, 577, 199, 604, 201, 58, 987, 511, 217, 654, 953, 930, 912, 854, 225, 142, 952, 166, 882, 801, 985, 905, 830, 298, 910, 448, 250, 719, 755, 996, 264, 928, 945, 601, 71, 289, 280, 639, 532, 436, 898, 303, 401, 948, 324, 674, 443, 846, 612, 825, 751, 545, 59, 638, 605, 766, 534, 382, 284, 531, 998, 454, 412, 947, 580, 449, 852, 322, 477, 965, 836, 796, 745, 538, 606, 926, 760, 713, 41, 967, 540, 994]
#b = {1: [1, 347, 421, 576, 599, 657, 873], 3: [42, 61, 683, 821], 6: [6, 70, 132, 150, 582], 13: [13, 62, 239, 321, 417, 609, 667, 723, 914, 989], 17: [17, 128, 215, 476, 753, 868, 988], 21: [21, 314, 390, 645, 755, 828, 841], 25: [2, 25, 207, 262, 551], 29: [29, 261, 306], 36: [36, 56, 155, 184, 292, 500, 642], 40: [28, 374, 635, 812], 41: [749, 903, 960], 49: [146, 299, 491], 50: [50, 345, 653, 810, 995], 57: [33, 57, 383, 890], 58: [211, 229, 646, 793], 59: [365, 607, 737], 63: [63, 241, 251, 381, 721], 64: [48, 502, 588, 768, 888], 71: [273, 294, 380, 400, 478, 934], 72: [78, 163, 341, 767], 75: [75, 145, 334, 650, 918], 76: [76, 453, 597], 77: [77, 109, 441, 999], 81: [81, 231, 556, 643, 791], 88: [73, 88, 156, 185, 194, 202, 203, 208, 255, 283, 293, 445, 457, 458, 473, 501, 568, 613, 692, 716, 727, 759, 775, 781, 803, 815, 881, 920, 927, 981], 99: [79, 99, 291, 571, 573, 693, 837], 113: [105, 113, 166, 183, 515, 569, 804, 917], 115: [44, 115, 785, 896], 119: [23, 55, 64, 119, 379, 483, 521, 743], 123: [123, 430, 679, 773], 124: [124, 247, 432, 438, 634], 125: [125, 169, 371], 126: [102, 126, 157, 798, 851, 925], 142: [226, 461, 566, 826, 932, 968], 148: [4, 162, 170, 178], 164: [164, 586, 757], 166: [232, 913], 176: [52, 176, 177, 319, 827, 863], 179: [114, 161, 253, 332, 848], 182: [3, 40, 101, 182, 575, 836], 190: [60, 190, 592, 805, 877, 943], 191: [22, 338, 688, 909], 193: [192, 193, 213, 252, 614], 198: [45, 254, 440, 797, 806, 887, 912], 199: [199, 201, 333, 625, 892, 894], 201: [209, 585, 707], 204: [74, 135, 204, 488, 682, 879], 206: [130, 206, 691, 843], 208: [32, 572, 923], 217: [217, 387, 520, 651, 786, 997, 998], 218: [66, 218, 429, 434, 442, 480, 704, 823], 220: [98, 220, 618, 784, 895], 225: [225, 542, 915], 229: [168, 355, 900, 942], 246: [15, 46, 242, 246, 259, 524], 250: [250, 375, 394, 490, 744], 253: [131, 159, 282, 318, 668], 257: [110, 257, 308], 264: [264, 466, 772], 280: [280, 370, 615, 970], 284: [389, 598, 830, 908], 286: [86, 89, 286, 407], 289: [276, 558, 787, 971], 291: [197, 747, 761, 779, 939], 298: [243, 298, 351, 564], 302: [100, 154, 302, 384, 624], 303: [303, 311, 719], 316: [153, 316, 325, 454, 543], 322: [446, 958], 323: [187, 205, 323, 328, 368, 517, 817, 938, 977], 324: [324, 369, 388, 959], 325: [35, 363, 392, 539, 581, 647, 681, 710, 729, 834], 330: [175, 330, 495, 690], 335: [95, 335, 731, 751], 336: [111, 336, 700, 740], 345: [195, 464, 471, 724], 352: [151, 352, 362, 591, 824, 961], 365: [0, 14, 59, 350, 396], 382: [382, 523, 975], 399: [94, 399, 406, 549, 706], 401: [310, 373, 401, 481, 526], 402: [152, 433, 448, 470, 584, 941], 412: [412, 512, 555], 416: [65, 322, 487, 548], 423: [69, 188, 216, 423], 425: [108, 189, 278, 281, 402, 425, 493, 503, 603, 660, 697], 431: [30, 80, 869, 951], 436: [296, 326, 327, 654], 443: [331, 443, 452, 687, 778, 948], 448: [248, 358, 413, 703, 818], 449: [431, 714, 800, 949], 451: [11, 118, 277, 295, 462, 829, 853], 454: [404, 820], 469: [117, 469, 619, 708, 754, 866, 963], 477: [477, 621, 765, 865, 907], 478: [58, 249, 420, 705], 482: [96, 472, 482, 902], 489: [72, 435, 489, 630], 501: [122, 782, 838, 916, 956, 978], 503: [160, 305, 637, 662, 762], 504: [90, 504, 633, 825], 507: [87, 91, 507, 578, 623], 509: [43, 233, 509, 537, 655, 735], 511: [214, 279, 511, 513, 540], 525: [158, 344, 525, 685], 527: [47, 116, 527, 589, 872, 947], 529: [12, 173, 403, 553, 911], 531: [391, 479, 531], 532: [290, 426, 532, 552, 565, 792, 811], 534: [378, 534, 629, 696, 904], 538: [538, 610, 972], 539: [37, 84, 342, 776, 835], 540: [799], 544: [53, 138, 307, 386, 410, 414, 415, 544, 924], 545: [364, 545, 844], 570: [5, 104, 354, 600], 577: [198, 315, 577, 715, 734, 885], 580: [419, 492, 580, 717, 846, 855], 590: [10, 107, 230, 357, 405, 499, 590, 670], 594: [186, 594], 595: [8, 140, 395, 422, 595, 649, 736, 746, 808], 601: [270, 317, 601], 604: [200, 210, 604, 627, 845, 940], 605: [376, 447, 494, 605, 763], 606: [606, 652, 860], 608: [34, 181, 301, 608, 648, 730], 611: [71, 304, 611, 789], 612: [343, 359, 508, 563, 612], 616: [16, 288, 616], 622: [180, 622, 864, 984], 628: [141, 143, 174, 628], 632: [18, 144, 287, 632, 756, 795], 638: [366, 638, 673, 780], 639: [285, 574, 639, 883], 640: [49, 275, 398, 640, 849, 992, 994], 650: [19, 309, 411, 726, 867, 886], 654: [219, 570, 884], 674: [329, 428, 674], 694: [133, 664, 694, 790, 878, 983], 705: [20, 26, 312, 519], 707: [106, 474, 485, 680], 713: [713, 794], 719: [256, 467, 701, 993], 728: [68, 516, 518, 579, 626, 722, 728], 739: [31, 93, 136, 147, 172, 272, 676, 739, 936], 742: [7, 228, 346, 408, 593, 742], 745: [536, 699, 745], 748: [134, 498, 546, 748, 944], 751: [360], 755: [258, 266, 456, 496, 931, 966], 760: [689, 695, 760], 766: [377, 529, 766, 847], 777: [112, 684, 769, 777, 807, 862, 882], 788: [41, 788], 796: [530, 567, 631, 796, 946], 801: [235, 636, 801, 899], 815: [54, 459, 672, 709], 819: [38, 196, 460, 535, 819], 825: [356, 562], 830: [240, 284, 533, 665, 858], 832: [24, 137, 510, 832, 897], 836: [514, 663, 725, 962], 837: [120, 397, 617, 822, 955], 844: [82, 149, 238], 846: [339, 348, 561, 666, 857], 848: [179, 289, 427, 468], 852: [436, 541, 852, 921, 980], 854: [224, 244, 353, 475, 854], 870: [171, 711, 758, 770, 870, 919], 879: [27, 455, 741, 974], 882: [234, 385, 698, 809], 891: [148, 165, 337, 340, 439, 465, 596, 671, 783, 875, 891, 901, 969, 976], 897: [67, 85, 260, 554, 602, 675, 764, 842], 898: [297, 409, 463, 547, 898], 905: [237, 269, 361, 905], 910: [245, 274, 814, 910], 912: [223, 450, 484, 712, 813, 935], 922: [9, 39, 449, 559, 583, 587, 644, 658, 720, 816, 859, 880, 922, 986], 926: [661, 926], 928: [267, 313, 560, 928, 964], 930: [222, 550, 930], 937: [129, 139, 372, 416, 738, 752, 937], 943: [92, 127, 802, 840], 945: [268, 702, 945], 947: [418], 948: [320, 367, 424, 451, 557, 669, 732, 750, 871, 876], 950: [83, 121, 437, 733, 950], 952: [227, 528, 771, 893, 952], 953: [221, 506, 718, 874, 929], 957: [167, 831, 957, 991], 965: [497, 505, 620, 656, 686, 965], 967: [774, 839, 967], 968: [51, 142, 444, 522, 906], 973: [103, 191, 486, 889, 973, 979], 985: [236, 861, 985], 987: [212, 271, 300, 678, 954, 987], 990: [97, 265, 933, 953, 990], 994: [982], 996: [263, 349, 677, 833, 856, 996], 998: [393, 641, 659, 850]}
#print(cvrp_global(20, a, b))

print(sam_apx3_w_ifs1(21, 50, 5))