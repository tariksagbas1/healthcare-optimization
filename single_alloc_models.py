from gurobipy import Model, GRB, quicksum
from math import dist, sqrt
from numpy import percentile, linspace, array
from time import time
from matplotlib import pyplot
from sklearn.cluster import KMeans

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
    """
    g_j = []
    for j in range(n):
        variables = 0.0
        for i in range(n):
            variables += b_ij[(i, j)]
        g_j.append(variables)
        model.addConstr(g_j[j] <= M * z_j[j])
    """
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
 - If two nodes are very far apart, it is unlikely that if a unit is placed on one of them, it will serve the other
 - If the distance between node i and j d_ij >= d_max, don't create variable b_ij
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

    all_distances = [d_ij[i][j] for i in range(n) for j in range(n) if i != j] ################## MODIFIED ####################
    d_max = percentile(all_distances, d_up) ################## MODIFIED ####################

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
        model.addConstr(variables <= C, name = f"capacity_constraint_{i}")
  
    # Population constraint
    for i in range(n):
        sum = 0.0
        for j in range(n):
            if b_ij[(i, j)]: ################## MODIFIED ####################
                sum += b_ij[(i, j)]
        model.addConstr(1 == sum, name=f"Population_constraint{i}")

    # Total unit constraint, sum(x_i) == m
    
    g_j = []
    for j in range(n):
        variables = 0.0
        for i in range(n):
            if b_ij[(i, j)]: ################## MODIFIED ####################
                variables += b_ij[(i, j)]
        g_j.append(variables)
        model.addConstr(g_j[j] <= M * z_j[j])

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
 - If p[i] is small, Z >= d_ij * b_ij * p doesn't need to be enforced. 
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

    all_distances = [d_ij[i][j] for i in range(n) for j in range(n) if i != j] ################## MODIFIED ####################
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
        model.addConstr(variables <= C, name = f"capacity_constraint_{i}")
  
    # Population constraint
    for i in range(n):
        sum = 0.0
        for j in range(n):
            if b_ij[(i, j)]: ################## MODIFIED ####################
                sum += b_ij[(i, j)]
        model.addConstr(1 == sum, name=f"Population_constraint{i}")

    # Total unit constraint, sum(x_i) == m
    
    g_j = []
    for j in range(n):
        variables = 0.0
        for i in range(n):
            if b_ij[(i, j)]: ################## MODIFIED ####################
                variables += b_ij[(i, j)]
        g_j.append(variables)
        model.addConstr(g_j[j] <= M * z_j[j])

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
 - Given pre-located units, solve it as assignment problem, use the solution as an ifs
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


    units = [67, 81, 71, 35, 37, 64, 92, 56, 33, 77, 52, 3, 66, 25, 16]
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

plot_function(7)