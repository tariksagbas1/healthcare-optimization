import sys
from gurobipy import Model, GRB, quicksum
from math import dist, sqrt
from numpy import percentile, linspace, array
from time import time
from matplotlib import pyplot
from sklearn.cluster import KMeans

"""
PLOTTING
"""
def plot_function(dataset_index, units = []):
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  
    p = []
    cord_com = []
    i = 1
    community_names = []
    for line in lines[2:]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, population = values[1], values[2], int(values[4])
        cord_com.append((x, y)) # add coordinates
        p.append(population)  # add populations for each community i
        community_names.append(i)
        i += 1

    colors = ['green' if name in units else 'black' for name in community_names]
    x, y = zip(*cord_com)
    pyplot.scatter(x, y, color=colors)
    pyplot.grid(True)

    
    for i, name in enumerate(community_names):
        pyplot.annotate(name, (x[i], y[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    
    pyplot.xticks(linspace(min(x), max(x), num=20))  # 20 intervals along the x-axis
    pyplot.yticks(linspace(min(y), max(y), num=20))  # 20 intervals along the y-axis    
    pyplot.show()

"""
GLOBAL OPTIMUM FINDER
"""
def g_model_global(dataset_index, n_p, C_p, m_p):


    # INITIAL DATA
    m = m_p  # Number of healthcare units to place
    C = C_p  # Capacity per healthcare unit
    n = n_p  # Number of communities
    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]


    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    for line in lines[2:(n+2)]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, population = values[1], values[2], int(values[4])
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

    # b_ij: amount of people in population i served by unit j
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype = GRB.INTEGER, name = f"b_{i}_{j}")

    # x_i: 1 if unit is placed on i'th community. 0 otherwise
    x_i = {}
    for i in range(n):
        x_i[i] = model.addVar(vtype = GRB.BINARY, name = f"x_{i}")

    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 
    
    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    # Z > b_ij * d_ij for every i, j
    for i in range(n):
        for j in range(n):
            model.addConstr(Z >= b_ij[i, j] * d_ij[i][j], name = f"Z_constraint_{i}_{j}")

    # Population constraint, sum(b_ij) == p_i for every population
    for i in range(n):
        model.addConstr(quicksum(b_ij[(i, j)] for j in range(n)) >= p[i], name = f"population_constraint_{i}")
    
    # Capacity constraint, sum(b_ij) <= x_i * C for every unit. (There can be surplus capacity ?)
    for i in range(n):
        variables = 0.0
        for j in range(n):
            variables += b_ij[(j, i)]
        model.addConstr(variables <= C * x_i[i], name = f"capacity_constraint_{i}")
  

    # Total unit constraint, sum(x_i) == m
    variables = 0.0
    for i in range(n):
        variables += x_i[i]
    model.addConstr(variables == m, name = "total unit constraint")

    # START TIMER
    start = time()

    # Solve Model
    model.optimize()
    print()
    # END TIMER
    end = time()
    time_elapsed = end - start
    if model.status == GRB.OPTIMAL:
        """for j in range(n):
            for i in range(n):
                if (i, j) in b_ij:
                    if b_ij[(i, j)].x != 0:
                        print(f"Unit on {j} sends {b_ij[(i, j)].x} items to community {i}")
                        """
        ls = []
        for i in range(len(x_i)):
            if x_i[i].x == 1:
                ls.append(i)
        print(ls)
        return str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x)
    else:
        return "Model Is Infeasible"

""""
FIRST METHOD OF APPROXIMATON:
 - If two nodes are very far apart, it is unlikely that if a unit is placed on one of them, it will serve the other
 - If the distance between node i and j d_ij >= d_max, don't create variable b_ij
"""
def g_model_apx1(dataset_index, n_p, C_p, m_p):


    # INITIAL DATA
    m = m_p  # Number of healthcare units to place
    C = C_p  # Capacity per healthcare unit
    n = n_p  # Number of communities
    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]


    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    for line in lines[2:(n+2)]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, population = values[1], values[2], int(values[4])
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
    d_max = percentile(all_distances, 80) ################## MODIFIED ####################
    # Create model

    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: amount of people in population i served by unit j
    b_ij = {}
    for i in range(n):
        for j in range(n):
            if d_ij[i][j] < d_max:  ################## MODIFIED ####################
                b_ij[(i, j)] = model.addVar(vtype = GRB.INTEGER, name = f"b_{i}_{j}")

    # x_i: 1 if unit is placed on i'th community. 0 otherwise
    x_i = {}
    for i in range(n):
        x_i[i] = model.addVar(vtype = GRB.BINARY, name = f"x_{i}")

    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 

    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    # Z > b_ij * d_ij for every i, j
    for i in range(n):
        for j in range(n):
            if i != j and (i, j) in b_ij: ################## MODIFIED ####################
                model.addConstr(Z >= b_ij[i, j] * d_ij[i][j], name = f"Z_constraint_{i}_{j}")

    # Population constraint, sum(b_ij) == p_i for every population
    for i in range(n):
        variables = 0.0
        for j in range(n):
            if (i, j) in b_ij: ################## MODIFIED ####################
                variables += b_ij[(i, j)]
        model.addConstr(variables == p[i], name = f"population_constraint_{i}")

    # Capacity constraint, sum(b_ij) <= x_i * C for every unit. (There can be surplus capacity ?)

    for i in range(n):
        variables = 0.0
        for j in range(n):
            if (j, i) in b_ij: ################## MODIFIED ####################
                variables += b_ij[(j, i)]
        model.addConstr(variables <= C * x_i[i], name = f"capacity_constraint_{i}")
  

    # Total unit constraint, sum(x_i) == m
    variables = 0.0
    for i in range(n):
        variables += x_i[i]
    model.addConstr(variables == m, name = "total unit constraint")

    # START TIMER
    start = time()

    # Solve Model
    model.optimize()

    # END TIMER
    end = time()
    time_elapsed = end - start
    if model.status == GRB.OPTIMAL:
        ls = []
        for i in range(n):
            if x_i[i].x == 1:
                ls.append(i)
        print(ls)
        return str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x)
    else:
        return "Model Is Infeasible"
    """
    if model.status == GRB.OPTIMAL:
        print("Healthcare Units Placed:")
        for i in range(n):
            if x_i[i].x == 1:
                print(f"Unit placed on community {i+1}, coordinates: {cord_com[i]}")
        print()
        for i in range(n):
            for j in range(n):
                if (i, j) in b_ij:
                    if b_ij[(i, j)].x != 0:
                        print(f"Service from {j+1} to {i+1}: {b_ij[(i, j)].x}")
        print()
        print(f"Z : {Z.x} \n")
        print(f"Total time: {time_elapsed:.2f} seconds \n")
    """

""""
NEW OBJECTIVE FUNCTION:
 min( sum(b_ij * d_ij) ) 
"""
def g_model_new_obj(dataset_index, n_p, C_p, m_p):




    # INITIAL DATA
    m = m_p  # Number of healthcare units to place
    C = C_p  # Capacity per healthcare unit
    n = n_p  # Number of communities
    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]


    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    for line in lines[2:(n+2)]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, population = values[1], values[2], int(values[4])
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

    d_max = 9999999

    # Create model
    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: amount of people in population i served by unit j
    b_ij = {}
    for i in range(n):
        for j in range(n):
            if d_ij[i][j] < d_max:  ################## MODIFIED ####################
                b_ij[(i, j)] = model.addVar(vtype = GRB.INTEGER, name = f"b_{i}_{j}")

    # x_i: 1 if unit is placed on i'th community. 0 otherwise
    x_i = {}
    for i in range(n):
        x_i[i] = model.addVar(vtype = GRB.BINARY, name = f"x_{i}")

    # Z: auxillary variable 
    obj = []
    for i in range(n):
        for j in range(n):
            obj.append(d_ij[i][j] * b_ij[(i, j)])



    Z = sum(obj)

    model.update()
    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints
    

    # Population constraint, sum(b_ij) == p_i for every population
    for i in range(n):
        variables = 0.0
        for j in range(n):
            if (i, j) in b_ij: ################## MODIFIED ####################
                variables += b_ij[(i, j)]
        model.addConstr(variables == p[i], name = f"population_constraint_{i}")

    # Capacity constraint, sum(b_ij) <= x_i * C for every unit. (There can be surplus capacity ?)
    for i in range(n):
        variables = 0.0
        for j in range(n):
            if (j, i) in b_ij: ################## MODIFIED ####################
                variables += b_ij[(j, i)]
        model.addConstr(variables <= C * x_i[i], name = f"capacity_constraint_{i}")

    # Total unit constraint, sum(x_i) == m
    variables = 0.0
    for i in range(n):
        variables += x_i[i]
    model.addConstr(variables == m, name = "total unit constraint")

    # START TIMER
    start = time()

    # Solve Model
    model.optimize()

    # END TIMER
    end = time()
    time_elapsed = end - start

    ls = []
    for i in range(n):
        for j in range(n):
            ls.append((b_ij[(i, j)].x) * d_ij[i][j])
    
    max_s = max(ls)

    if model.status == GRB.OPTIMAL:
        return str(round(time_elapsed, 2)) + " seconds, Z : " + str(max_s)
    else:
        return "Model Is Infeasible"

"""
SECOND METHOD OF APPROXIMATION
 - All the methods of First Method of Approximation
 - If two nodes are very close, then if a unit is placed on node j, service sent from j to i, b_ij times d_ij must be low.
 - If distance is low, don't add the constraint Z >= b_ij * d_ij
"""
def g_model_apx2(dataset_index, n_p, C_p, m_p):




    # INITIAL DATA
    m = m_p  # Number of healthcare units to place
    C = C_p  # Capacity per healthcare unit
    n = n_p  # Number of communities
    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]


    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    for line in lines[2:(n+2)]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, population = values[1], values[2], int(values[4])
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
    d_max = percentile(all_distances, 80) ################## MODIFIED ####################
    d_min = percentile(all_distances, 10) ################## MODIFIED ####################

    # Create model
    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: amount of people in population i served by unit j
    b_ij = {}
    for i in range(n):
        for j in range(n):
            if d_ij[i][j] < d_max:  ################## MODIFIED ####################
                b_ij[(i, j)] = model.addVar(vtype = GRB.INTEGER, name = f"b_{i}_{j}")

    # x_i: 1 if unit is placed on i'th community. 0 otherwise
    x_i = {}
    for i in range(n):
        x_i[i] = model.addVar(vtype = GRB.BINARY, name = f"x_{i}")

    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 

    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    # Z > b_ij * d_ij for every i, j
    for i in range(n):
        for j in range(n):
            if i != j and (i, j) in b_ij and d_ij[i][j] >= d_min: ################## MODIFIED ####################
                model.addConstr(Z >= b_ij[i, j] * d_ij[i][j], name = f"Z_constraint_{i}_{j}")

    # Population constraint, sum(b_ij) == p_i for every population
    for i in range(n):
        variables = 0.0
        for j in range(n):
            if (i, j) in b_ij: ################## MODIFIED ####################
                variables += b_ij[(i, j)]
        model.addConstr(variables == p[i], name = f"population_constraint_{i}")

    # Capacity constraint, sum(b_ij) <= x_i * C for every unit. (There can be surplus capacity ?)
    for i in range(n):
        variables = 0.0
        for j in range(n):
            if (j, i) in b_ij: ################## MODIFIED ####################
                variables += b_ij[(j, i)]
        model.addConstr(variables <= C * x_i[i], name = f"capacity_constraint_{i}")

    # Total unit constraint, sum(x_i) == m
    variables = 0.0
    for i in range(n):
        variables += x_i[i]
    model.addConstr(variables == m, name = "total unit constraint")

    # START TIMER
    start = time()

    # Solve Model
    model.optimize()

    # END TIMER
    end = time()
    time_elapsed = end - start
    if model.status == GRB.OPTIMAL:
        return str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x)
    else:
        return "Model Is Infeasible"
"""
THIRD METHOD OF APPROXIMATION
 - All the methods of Second Method of Approximation
 - If two nodes are very close and the i'th nodes population is not very high, then it is unlikely that if a unit is placed on j, b_ij(service) * d_ij(distance) is a high number .
 - If distance and population is low, don't add the constraint Z >= b_ij * d_ij
"""
def g_model_apx3(dataset_index, n_p, C_p, m_p):




    # INITIAL DATA
    m = m_p  # Number of healthcare units to place
    C = C_p  # Capacity per healthcare unit
    n = n_p  # Number of communities
    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]


    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    for line in lines[2:(n+2)]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, population = values[1], values[2], int(values[4])
        cord_com.append([x, y]) # add coordinates
        p.append(population)  # add populations for each community i

    p_max = float(percentile(p, 90))



    # Calculate distances
    d_ij = []
    for x1, y1 in cord_com:
        row = []
        for x2, y2 in cord_com:
            distance = dist((x1, y1), (x2, y2))
            row.append(distance)
        d_ij.append(row)

    all_distances = [d_ij[i][j] for i in range(n) for j in range(n) if i != j] ################## MODIFIED ####################
    d_max = percentile(all_distances, 90) ################## MODIFIED ####################
    d_min = percentile(all_distances, 10) ################## MODIFIED ####################

    # Create model
    model = Model("Healthcare Placement")

    # Decision variables

    # b_ij: amount of people in population i served by unit j
    b_ij = {}
    for i in range(n):
        for j in range(n):
            if d_ij[i][j] < d_max:  ################## MODIFIED ####################
                b_ij[(i, j)] = model.addVar(vtype = GRB.INTEGER, name = f"b_{i}_{j}")

    # x_i: 1 if unit is placed on i'th community. 0 otherwise
    x_i = {}
    for i in range(n):
        x_i[i] = model.addVar(vtype = GRB.BINARY, name = f"x_{i}")

    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 

    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    # Z > b_ij * d_ij for every i, j
    for i in range(n):
        for j in range(n):
            if i != j and (i, j) in b_ij and d_ij[i][j] >= d_min and p[i] <= p_max: ################## MODIFIED ####################
                model.addConstr(Z >= b_ij[i, j] * d_ij[i][j], name = f"Z_constraint_{i}_{j}")

    # Population constraint, sum(b_ij) == p_i for every population
    for i in range(n):
        variables = 0.0
        for j in range(n):
            if (i, j) in b_ij: ################## MODIFIED ####################
                variables += b_ij[(i, j)]
        model.addConstr(variables == p[i], name = f"population_constraint_{i}")

    # Capacity constraint, sum(b_ij) <= x_i * C for every unit. (There can be surplus capacity ?)
    for i in range(n):
        variables = 0.0
        for j in range(n):
            if (j, i) in b_ij: ################## MODIFIED ####################
                variables += b_ij[(j, i)]
        model.addConstr(variables <= C * x_i[i], name = f"capacity_constraint_{i}")

    # Total unit constraint, sum(x_i) == m
    variables = 0.0
    for i in range(n):
        variables += x_i[i]
    model.addConstr(variables == m, name = "total unit constraint")

    # START TIMER
    start = time()

    # Solve Model
    model.optimize()

    # END TIMER
    end = time()
    time_elapsed = end - start
    if model.status == GRB.OPTIMAL:
        return str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x)
    else:
        return "Model Is Infeasible"
"""
ITERATIVE METHOD 1
 - Start with r constraints, solve model and add the most violated constraint at every iteration and solve again
 - Since there are many redundant >= 0 constraints, avoid as many as possible by solving iteratively
 - (The constraint refferred to here are Z >= constraints)
"""
def g_model_iterative(dataset_index, n_p, C_p, m_p, r):
    # INITIAL DATA
    m = m_p  # Number of healthcare units to place
    C = C_p  # Capacity per healthcare unit
    n = n_p  # Number of communities
    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]

    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    for line in lines[2:(n+2)]:  # skip first two lines
        values = list(map(float, line.split())) # Convert all values to floats
        x, y, population = values[1], values[2], int(values[4])
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
    
    model = Model("Healthcare Placement")

    # Decision Variables

    # b_ij: amount of people in population i served by unit j
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype = GRB.INTEGER, name = f"b_{i}_{j}")
    
    # x_i: 1 if unit is placed on i'th community. 0 otherwise
    x_i = {}
    for i in range(n):
        x_i[i] = model.addVar(vtype = GRB.BINARY, name = f"x_{i}")

    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 

    model.update()
    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    # Population constraint, sum(b_ij) == p_i for every population
    for i in range(n):
        variables = 0.0
        for j in range(n):
            variables += b_ij[(i, j)]
        model.addConstr(variables == p[i], name = f"population_constraint_{i}")

    # Capacity constraint, sum(b_ij) <= x_i * C for every unit. (There can be surplus capacity ?)
    for i in range(n):
        variables = 0.0
        for j in range(n):
            variables += b_ij[(j, i)]
        model.addConstr(variables <= C * x_i[i], name = f"capacity_constraint_{i}")

    # Total unit constraint, sum(x_i) == m
    variables = 0.0
    for i in range(n):
        variables += x_i[i]
    model.addConstr(variables == m, name = "total unit constraint")

    # Z > b_ij * d_ij for every i, j
    for i in range(int(sqrt(r))):
        for j in range(int(sqrt(r))):
            model.addConstr(Z >= b_ij[i, j] * d_ij[i][j], name = f"Z_constraint_{i}_{j}")
    h = 0

    # Start Timer
    start = time()

    
    for c in range((n * n) - r):
        
        # Solve Model
        model.optimize()
        z_val = Z.x

        # Solve model, determine the constraints, add the most violated constraint
        dict = {}
        for i in range(n):
            for j in range(n):
                if b_ij[(i, j)].x != 0:
                    dict[(i, j)] = b_ij[(i, j)].x * d_ij[i][j]
        
        
        lister = sorted(dict.items(), key=lambda item: item[1], reverse=True) # Sort the list of constraints
        max_cons_tuple = lister[0][0]
        max_cons_val = lister[0][1]
        if z_val < max_cons_val: 
            c_index = max_cons_tuple[0] # Get the max constraints i'th index 
            u_index = max_cons_tuple[1] # Get the max constraints j'th index
            model.addConstr(Z >= b_ij[c_index, u_index] * d_ij[c_index][u_index], name = f"Z_constraint_{c_index}_{u_index}")
            model.addConstr(Z >= b_ij[lister[1][0][0], lister[1][0][1]] * d_ij[lister[1][0][0]][lister[1][0][1]], name = f"Z_constraint_{lister[1][0][0]}_{lister[1][0][1]}")
            model.addConstr(Z >= b_ij[lister[2][0][0], lister[2][0][1]] * d_ij[lister[2][0][0]][lister[2][0][1]], name = f"Z_constraint_{lister[2][0][0]}_{lister[2][0][1]}")
            model.addConstr(Z >= b_ij[lister[3][0][0], lister[3][0][1]] * d_ij[lister[3][0][0]][lister[3][0][1]], name = f"Z_constraint_{lister[3][0][0]}_{lister[3][0][1]}")
            model.addConstr(Z >= b_ij[lister[4][0][0], lister[4][0][1]] * d_ij[lister[4][0][0]][lister[4][0][1]], name = f"Z_constraint_{lister[4][0][0]}_{lister[4][0][1]}")
            model.addConstr(Z >= b_ij[lister[5][0][0], lister[5][0][1]] * d_ij[lister[5][0][0]][lister[5][0][1]], name = f"Z_constraint_{lister[5][0][0]}_{lister[5][0][1]}")
            model.addConstr(Z >= b_ij[lister[6][0][0], lister[6][0][1]] * d_ij[lister[6][0][0]][lister[6][0][1]], name = f"Z_constraint_{lister[6][0][0]}_{lister[6][0][1]}")
            h += 7
        else:
            break
    
    # END TIMER
    end = time()
    time_elapsed = end - start
       
    if model.status == GRB.OPTIMAL:
        print(f"Z : {Z.x} \n")
        print(f"Total time: {time_elapsed:.2f} seconds \n")
        print(f"Total constraints avoided: {n*n - h}")
    else:
        return "Model Is Infeasible"
"""
ITERATIVE METHOD 2
 - All methods in iterative method 1
 - Select the starting r constraints according to the largest r distances.
 - A better starting point
"""
def g_model_iterative2(dataset_index, n_p, C_p, m_p, r):
    # INITIAL DATA
    m = m_p  # Number of healthcare units to place
    C = C_p  # Capacity per healthcare unit
    n = n_p  # Number of communities
    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]


    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    for line in lines[2:(n+2)]:  # skip first two lines
        values = list(map(float, line.split())) # Convert all values to floats
        x, y, population = values[1], values[2], int(values[4])
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

    asd = []
    for i in range(n):
        for j in range(n):
            tup = (i, j, d_ij)
            asd.append(tup)
    
    largest_r_ = sorted(asd, reverse=True, key= lambda el: el[2])[:r]
    
    largest_r = []
    for i in range(r):
        tup = (largest_r_[i][0], largest_r_[i][1])
        largest_r.append(tup)
    
    model = Model("Healthcare Placement")
    model.setParam(GRB.Param.MIPGap, 0.05)

    # Decision Variables

    # b_ij: amount of people in population i served by unit j
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype = GRB.INTEGER, name = f"b_{i}_{j}")
    
    # x_i: 1 if unit is placed on i'th community. 0 otherwise
    x_i = {}
    for i in range(n):
        x_i[i] = model.addVar(vtype = GRB.BINARY, name = f"x_{i}")

    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 

    model.update()
    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    # Population constraint, sum(b_ij) == p_i for every population
    for i in range(n):
        variables = 0.0
        for j in range(n):
            variables += b_ij[(i, j)]
        model.addConstr(variables == p[i], name = f"population_constraint_{i}")

    # Capacity constraint, sum(b_ij) <= x_i * C for every unit. (There can be surplus capacity ?)
    for i in range(n):
        variables = 0.0
        for j in range(n):
            variables += b_ij[(j, i)]
        model.addConstr(variables <= C * x_i[i], name = f"capacity_constraint_{i}")

    # Total unit constraint, sum(x_i) == m
    variables = 0.0
    for i in range(n):
        variables += x_i[i]
    model.addConstr(variables == m, name = "total unit constraint")

    # Z > b_ij * d_ij for every i, j
    for i in range(n):
        for j in range(n):
            if (i,j) in largest_r:
                model.addConstr(Z >= b_ij[i, j] * d_ij[i][j], name = f"Z_constraint_{i}_{j}")
    h = 0

    # Start Timer
    start = time()

    for c in range((n * n) - r):
    
        # Solve Model
        model.optimize()
        z_val = Z.x

        # Solve model, determine the constraints, add the most violated constraint
        dict = {}
        for i in range(n):
            for j in range(n):
                if b_ij[(i, j)].x != 0:
                    dict[(i, j)] = b_ij[(i, j)].x * d_ij[i][j]
        
        
        lister = sorted(dict.items(), key=lambda item: item[1], reverse=True) # Sort the list of constraints
        max_cons_tuple = lister[0][0]
        max_cons_val = lister[0][1]
        if z_val < max_cons_val: 
            c_index = max_cons_tuple[0] # Get the max constraints i'th index 
            u_index = max_cons_tuple[1] # Get the max constraints j'th index
            model.addConstr(Z >= b_ij[c_index, u_index] * d_ij[c_index][u_index], name = f"Z_constraint_{c_index}_{u_index}")
            model.addConstr(Z >= b_ij[lister[1][0][0], lister[1][0][1]] * d_ij[lister[1][0][0]][lister[1][0][1]], name = f"Z_constraint_{lister[1][0][0]}_{lister[1][0][1]}")
            model.addConstr(Z >= b_ij[lister[2][0][0], lister[2][0][1]] * d_ij[lister[2][0][0]][lister[2][0][1]], name = f"Z_constraint_{lister[2][0][0]}_{lister[2][0][1]}")
            model.addConstr(Z >= b_ij[lister[3][0][0], lister[3][0][1]] * d_ij[lister[3][0][0]][lister[3][0][1]], name = f"Z_constraint_{lister[3][0][0]}_{lister[3][0][1]}")
            model.addConstr(Z >= b_ij[lister[4][0][0], lister[4][0][1]] * d_ij[lister[4][0][0]][lister[4][0][1]], name = f"Z_constraint_{lister[4][0][0]}_{lister[4][0][1]}")
            model.addConstr(Z >= b_ij[lister[5][0][0], lister[5][0][1]] * d_ij[lister[5][0][0]][lister[5][0][1]], name = f"Z_constraint_{lister[5][0][0]}_{lister[5][0][1]}")
            model.addConstr(Z >= b_ij[lister[6][0][0], lister[6][0][1]] * d_ij[lister[6][0][0]][lister[6][0][1]], name = f"Z_constraint_{lister[6][0][0]}_{lister[6][0][1]}")
            h += 7
        else:
            break
    
    # END TIMER
    end = time()
    time_elapsed = end - start
       
    if model.status == GRB.OPTIMAL:
        print("Healthcare Units Placed:")
        for i in range(n):
            if x_i[i].x == 1:
                print(f"Unit placed on community {i+1}, coordinates: {cord_com[i]}")
        print()
        for i in range(n):
            for j in range(n):
                if b_ij[(i, j)].x != 0:
                    #print(f"Service from {j+1} to {i+1}: {b_ij[(i, j)].x}")
                    print(f"Z value for {j + 1} to {i + 1}: {b_ij[(i, j)].x * d_ij[i][j]} ")
        print()
        print(f"Z : {Z.x} \n")
        print(f"Total time: {time_elapsed:.2f} seconds \n")
        print(f"Total constraints avoided: {n*n - h}")
    else:
        return "Model Is Infeasible"

"""
CLUSTERING METHOD
 - Use of clustering heuristic to place units.
"""
def g_model_clustering(dataset_index, n_p, C_p, m_p):
    # INITIAL DATA
    m = m_p  # Number of healthcare units to place
    C = C_p  # Capacity per healthcare unit
    n = n_p  # Number of communities
    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]

    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    for line in lines[2:(n+2)]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, population = values[1], values[2], int(values[4])
        cord_com.append((x, y)) # add coordinates
        p.append(population)  # add populations for each community i

    # Calculate distances
    d_ij = []
    for x1, y1 in cord_com:
        row = []
        for x2, y2 in cord_com:
            distance = dist((x1, y1), (x2, y2))
            row.append(distance)
        d_ij.append(row)

    abc = []
    for i in range(n):
        d_list = []
        for j in range(n):
            d_list.append(d_ij[i][j] * p[j])
        sum_of_min_weighted_distances = sum(sorted(d_list)[0:5])
        abc.append((i + 1, sum_of_min_weighted_distances))
    
    res = sorted(abc, key = lambda tup: tup[1])
    units = []
    for i,j in res[:m]:
        units.append(i)

    

    # Create model
    model = Model("Healthcare Placement")
    
    # Decision variables

    # b_ij: amount of people in population i served by unit j
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype = GRB.INTEGER, name = f"b_{i}_{j}")

    # x_i: 1 if unit is placed on i'th community. 0 otherwise
    x_i = {}
    for i in range(n):
        x_i[i] = model.addVar(vtype = GRB.BINARY, name = f"x_{i}")

    for i in range(n):
        if i in units:
            x_i[i].Start = 1
    
    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 

    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints

    # Z > b_ij * d_ij for every i, j
    for i in range(n):
        for j in range(n):
            model.addConstr(Z >= b_ij[i, j] * d_ij[i][j], name = f"Z_constraint_{i}_{j}")

    # Population constraint, sum(b_ij) == p_i for every population
    for i in range(n):
        variables = 0.0
        for j in range(n):
            variables += b_ij[(i, j)]
        model.addConstr(variables == p[i], name = f"population_constraint_{i}")

    # Capacity constraint, sum(b_ij) <= x_i * C for every unit. (There can be surplus capacity ?)

    for i in range(n):
        variables = 0.0
        for j in range(n):
            variables += b_ij[(j, i)]
        model.addConstr(variables <= C * x_i[i], name = f"capacity_constraint_{i}")
  

    # Total unit constraint, sum(x_i) == m
    variables = 0.0
    for i in range(n):
        variables += x_i[i]
    model.addConstr(variables == m, name = "total unit constraint")

    # START TIMER
    start = time()

    # Solve Model
    model.optimize()
    print()
    # END TIMER
    end = time()
    time_elapsed = end - start
    if model.status == GRB.OPTIMAL:
        ls = []
        for i in range(n):
            if x_i[i].x == 1:
                ls.append(i)
        print(ls)
        return str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x)
    else:
        return "Model Is Infeasible"
"""
INITIAL FEASIBLE SOLUTION
 - Generation of ifs with assignment problem
 - Given pre-located units, solve it as assignment problem, use the solution as an ifs
"""
def ifs1(dataset_index, n_p, C_p, m_p, units: list):


    # INITIAL DATA
    m = m_p  # Number of healthcare units to place
    C = C_p  # Capacity per healthcare unit
    n = n_p # Number of communities
    p = [] # Population of every community
    units = units # Units placed on nodes
    cord_com = [] # Coordinate of every population [x, y]


    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    for line in lines[2:(n+2)]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, population = values[1], values[2], int(values[4])
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
    u = {}
    for unit_index in units:
        for i in range(n):
            for j in range(C):
                u[(i, unit_index, j)] = model.addVar(vtype = GRB.BINARY, name = f"u_{i}_{unit_index}_{j}")
            


    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 
    
    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Constraints


    # Z constraint
    for i in range(n):
        for j in units:
            b_ij = 0
            for k in range(C):
                b_ij += u[(i, j, k)]
            
            model.addConstr(Z >= b_ij * d_ij[i][j], name = f"Z_constraint_{i}_{j}")

    # Population constraint, sum(b_ij) == p_i for every population
    for i in range(n):
        variables = 0.0
        for unit_index in units:
            for j in range(C):
                variables += u[(i, unit_index, j)]
        model.addConstr(p[i] <= variables, name = f"population_constraint_{i}")

    # Capacity constraint
    
    for unit_index in units:
        for j in range(C):
            variables = 0.0
            for i in range(n):
                variables += u[(i, unit_index, j)]
            model.addConstr(variables <= 1, name = f"assignment_constraint_{j}_{unit_index}")
    
    
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
        print(f"IFS took: {str(round(time_elapsed, 2))}  seconds, Z : {str(Z.x)}")
        
        b_ij = {}
        x_i = []
        for i in range(n):
            for j in range(n):
                bij = 0
                if j in units:
                    for k in range(C):
                        bij += u[(i, j, k)].x
                    b_ij[(i, j)] = bij
                else:
                    b_ij[(i, j)] = 0
        
        for j in range(n):
            if j in units:
                x_i.append(1)
            else:
                x_i.append(0)

        return Z.x, b_ij, x_i, round(time_elapsed, 2)
    else:
        return "Model Is Infeasible"

def g_model_with_ifs1(dataset_index, n_p, C_p, m_p):
    # INITIAL DATA
    m = m_p  # Number of healthcare units to place
    C = C_p  # Capacity per healthcare unit
    n = n_p  # Number of communities
    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]

    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    for line in lines[2:(n+2)]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, population = values[1], values[2], int(values[4])
        cord_com.append((x, y)) # add coordinates
        p.append(population)  # add populations for each community i

    # Calculate distances
    d_ij = []
    for x1, y1 in cord_com:
        row = []
        for x2, y2 in cord_com:
            distance = dist((x1, y1), (x2, y2))
            row.append(distance)
        d_ij.append(row)

    abc = []
    for i in range(n):
        d_list = []
        for j in range(n):
            d_list.append(d_ij[i][j] * p[j])
        sum_of_min_weighted_distances = sum(sorted(d_list)[0:5])
        abc.append((i + 1, sum_of_min_weighted_distances))
    
    res = sorted(abc, key = lambda tup: tup[1])
    units = []
    for i,j in res[:m]:
        units.append(i)

    

    # Create model
    model = Model("Healthcare Placement")
    
    # Decision variables

    # b_ij: amount of people in population i served by unit j
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = model.addVar(vtype = GRB.INTEGER, name = f"b_{i}_{j}")

    # x_i: 1 if unit is placed on i'th community. 0 otherwise
    x_i = {}
    for i in range(n):
        x_i[i] = model.addVar(vtype = GRB.BINARY, name = f"x_{i}")
    
    # Z: auxillary variable 
    Z = model.addVar(vtype = GRB.CONTINUOUS, name = "Z") 

    model.update()

    model.setObjective(Z, GRB.MINIMIZE)

    # Start with IFS

    Z_start, b_ij_start, x_i_start, ifs_time  = ifs1(dataset_index, n, C, m, units)

    for i in range(n):
        x_i[i].start = x_i_start[i]
        for j in range(n):
            b_ij[(i, j)].start = b_ij_start[(i, j)]
    
    # Constraints

    # Z > b_ij * d_ij for every i, j
    for i in range(n):
        for j in range(n):
            model.addConstr(Z >= b_ij[i, j] * d_ij[i][j], name = f"Z_constraint_{i}_{j}")

    # Population constraint, sum(b_ij) == p_i for every population
    for i in range(n):
        variables = 0.0
        for j in range(n):
            variables += b_ij[(i, j)]
        model.addConstr(variables == p[i], name = f"population_constraint_{i}")

    # Capacity constraint, sum(b_ij) <= x_i * C for every unit. (There can be surplus capacity ?)

    for i in range(n):
        variables = 0.0
        for j in range(n):
            variables += b_ij[(j, i)]
        model.addConstr(variables <= C * x_i[i], name = f"capacity_constraint_{i}")
  

    # Total unit constraint, sum(x_i) == m
    variables = 0.0
    for i in range(n):
        variables += x_i[i]
    model.addConstr(variables == m, name = "total unit constraint")

    # START TIMER
    start = time()

    # Solve Model
    model.optimize()
    print()
    # END TIMER
    end = time()
    time_elapsed = end - start
    if model.status == GRB.OPTIMAL:
        print()
        print(f"IFS took: {str(round(time_elapsed, 2))}, Z-ifs: {Z_start}")
        return str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x)
    else:
        return "Model Is Infeasible"

def g_new(dataset_index, n_p, C_p, m_p):

    M = n_p
    # INITIAL DATA
    m = m_p  # Number of healthcare units to place
    C = C_p  # Capacity per healthcare unit
    n = n_p  # Number of communities
    p = [] # Population of every community
    cord_com = [] # Coordinate of every population [x, y]


    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()  

    for line in lines[2:(n+2)]:  # skip first two lines
        values = list(map(float, line.split()))  # Convert all values to floats
        x, y, population = values[1], values[2], int(values[4])
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
    
    # Capacity constraint, sum(b_ij) <= x_i * C for every unit. (There can be surplus capacity ?)
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

    # Solve Model
    model.optimize()
    print()
    # END TIMER
    end = time()
    time_elapsed = end - start
    if model.status == GRB.OPTIMAL:
        for j in range(n):
            for i in range(n):
                if (i, j) in b_ij:
                    if b_ij[(i, j)].x != 0:
                        print(f"Unit on {j} sends to community {i}")
                        
        ls = []
        for i in range(n):
            for j in range(n):
                if b_ij[(i, j)].x == 1 and j not in ls:
                    ls.append(j)
        print(ls)
        return str(round(time_elapsed, 2)) + " seconds, Z : " + str(Z.x)
    else:
        return "Model Is Infeasible"
    
print(g_new(2, 100, 321, 15))