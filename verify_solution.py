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

# Example usage:
ok, result = find_any_feasible(8)
if ok:
    print("Feasible configuration found!")
    print(result)
else:
    print("Infeasible:", result)

