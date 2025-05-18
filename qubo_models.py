from math import dist
from time import time
from dimod import BinaryQuadraticModel
import dimod
from gurobipy import Model, GRB, quicksum

def qubo_sam_global(dataset_index):

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

    bqm = BinaryQuadraticModel.empty(vartype = dimod.BINARY)

    # Decision Variables
    b_ij = {}
    for i in range(n):
        for j in range(n):
            b_ij[(i, j)] = f"b_{i}{j}"
            bqm.add_variable(f"b_{i}{j}", 0)

    z_i = {}
    for i in range(n):
        z_i[i] = f"z_{i}"
        bqm.add_variable(f"z_{i}", 0)

    # Z = sum( (2 ** k) * Z_bit_k ) for k = 0 1 2 ... 15
    Z_bits = {}
    for i in range(16):
        Z_bits[i] = f"Z_bit_{i}"
        bqm.add_variable(f"Z_bit_{i}", 0)

