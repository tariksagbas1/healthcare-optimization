from math import dist
from single_alloc_models import d_cluster_new
import os
from itertools import combinations

def load_instance(dataset_index):
    """Load populations and distance matrix from Instance_{dataset_index}.txt."""
    p = []
    cord_com = []
    path = os.path.join("datasets", f"Instance_{dataset_index}.txt")
    with open(path, "r") as f:
        lines = f.readlines()
    # skip header lines
    for line in lines[2:]:
        vals = list(map(float, line.split()))
        x, y, _, pop = vals[1], vals[2], vals[3], int(vals[4])
        cord_com.append((x, y))
        p.append(pop)
    # build full distance dict
    d_ij = {}
    for i, (x1, y1) in enumerate(cord_com):
        for j, (x2, y2) in enumerate(cord_com):
            d_ij[(i, j)] = dist((x1, y1), (x2, y2))
    return p, d_ij

def verify_solution(dataset_index, assignments, alpha, beta):
    """
    assignments: dict mapping unit_index -> list of community indices
    alpha: workload threshold
    beta: distance threshold
    Returns (alpha_ok, beta_ok, details_dict)
    """
    p, d_ij = load_instance(dataset_index)

    # 1) workloads per unit
    workloads = {u: sum(p[i] for i in comms)
                 for u, comms in assignments.items()}
    max_w, min_w = max(workloads.values()), min(workloads.values())
    alpha_ok = (max_w - min_w) <= alpha

    # 2) distances walked per community
    dist_walked = {}
    for u, comms in assignments.items():
        for i in comms:
            dist_walked[i] = d_ij[(i, u)]
    max_d, min_d = max(dist_walked.values()), min(dist_walked.values())
    beta_ok = (max_d - min_d) <= beta

    details = {
        "max_w": max_w, "min_w": min_w,
        "workload_spread": max_w - min_w,
        "alpha": alpha, "alpha_ok": alpha_ok,
        "max_d": max_d, "min_d": min_d,
        "distance_spread": max_d - min_d,
        "beta": beta, "beta_ok": beta_ok
    }
    return alpha_ok and beta_ok



def find_any_feasible(dataset_index):
    """
    Tries every choice of M facility locations (i.e. indices in 0..n-1)
    and returns the first one that satisfies capacity, alpha‐ and beta‐spread.
    """
    p, C, coords, M = load_instance(dataset_index)
    n = len(p)

    # precompute full distance matrix
    d_ij = {
        (i, j): dist(coords[i], coords[j])
        for i in range(n) for j in range(n)
    }

    # thresholds
    alpha = round((sum(p)/M)*0.2)
    beta  = max(d_ij.values())*0.2

    # trivial global capacity check
    if sum(p) > M*C:
        return False, "Total demand exceeds total capacity"

    # trivial alpha check
    if max(p) > alpha:
        return False, f"Single community pop {max(p)} > α = {alpha}"

    # try every combination of M distinct units
    for units in combinations(range(n), M):
        # compute nearest‐facility distance for each community
        nearest = [min(d_ij[(i,u)] for u in units) for i in range(n)]
        spread = max(nearest) - min(nearest)
        if spread <= beta:
            # found a feasible set
            return True, {
                "units":      units,
                "alpha":      alpha,
                "beta":       beta,
                "capacity":   C,
                "max_d":      max(nearest),
                "min_d":      min(nearest),
                "spread_d":   spread,
            }

    return False, "No choice of units meets the β‐spread constraint"

print(verify_solution(12, {0: [0, 29, 59, 91, 110], 1: [1, 32, 42, 92, 116, 220], 7: [7, 166, 183, 273], 9: [44, 152, 271], 12: [19, 65, 83, 87, 217, 239], 13: [96, 207, 226, 254, 289], 33: [12, 126, 132, 169, 172], 44: [2, 21, 79, 128, 175, 222, 268, 284], 46: [46, 62, 118, 186, 285], 48: [38, 69, 93, 170, 199, 221, 235, 237], 54: [20, 49, 54, 122, 181, 190, 200, 214], 55: [55, 90, 121, 184, 288], 56: [15, 23, 39, 56, 134], 66: [112, 114, 130, 204, 212, 246], 67: [33, 35, 70, 194, 203, 205, 238], 69: [34, 51, 68, 188, 209], 94: [5, 123, 135, 171, 202, 247], 96: [14, 43, 139, 195, 294], 101: [27, 164, 165, 173, 174, 178, 270, 272], 102: [113, 198, 228, 242, 274, 278], 117: [30, 117, 119, 142, 167, 245, 283], 128: [52, 133, 255, 267, 276, 281], 134: [18, 26, 146, 180], 135: [24, 36, 47, 136, 231, 243, 260], 148: [3, 74, 94, 148, 244], 150: [125, 150, 158, 196], 151: [28, 66, 97, 101, 151], 159: [61, 82, 208, 215, 297], 162: [108, 137, 147, 162, 292, 296], 163: [57, 81, 127, 163, 275, 286], 182: [143, 153, 182, 236, 263, 279], 185: [10, 80, 138, 168, 185, 210], 197: [11, 140, 192, 197, 206], 213: [89, 115, 157, 234, 280], 215: [76, 131, 154, 159], 223: [85, 100, 155, 193, 223, 230], 225: [67, 109, 213, 240, 287], 229: [17, 41, 72, 120, 144, 161], 232: [31, 63, 75, 252, 266], 233: [8, 86, 227, 233, 249, 257, 269], 234: [50, 191, 218, 219], 235: [22, 40, 48, 78, 129, 299], 240: [37, 53, 77, 105, 225, 256], 258: [98, 104, 258, 262, 291], 261: [64, 88, 99, 107, 211, 261, 282], 265: [16, 60, 149, 259, 265], 266: [6, 58, 111, 141, 187, 232, 290], 272: [45, 124, 145, 179, 251], 273: [4, 84, 106, 177, 224, 250, 293], 277: [73, 103, 248, 277], 278: [9, 71, 102, 160, 229], 283: [13, 25, 176, 189, 216], 292: [95, 156, 253, 264], 298: [201, 241, 295, 298]}, 55 ,917.0832241405357))