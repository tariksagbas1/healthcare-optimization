from gurobipy import Model, GRB
from math import dist
"""
CAPACITATED VEHICLE ROUTING PROBLEM
"""
def cvrp_global(dataset_index, units = list(), assignments = dict()):
    full_units = [0] + units
    unit_cords = [] # Healthcare unit coordinates
    q = {} # Unit equipment need
    p = [] # Population of every community

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

    total = sum(p)
    M = 5
    if M * 10000 < total:
        return "Infeasible"
    
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
    model.setParam("PumpPasses", 50)
    model.setParam("MIPFocus", 1)

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