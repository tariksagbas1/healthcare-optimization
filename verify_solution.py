from math import dist
import os

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

print(verify_solution(7, {4: [53, 124, 187, 192, 195], 6: [37, 46, 61, 150, 155, 182], 9: [2, 63, 89, 118, 136, 141], 10: [14, 58, 74, 83, 129], 21: [8, 21, 96, 100, 115, 119], 22: [105, 122, 128, 170], 25: [15, 47, 88, 148, 158, 177], 30: [3, 24, 38, 40, 145], 31: [6, 20, 31, 69, 106, 140, 180], 35: [22, 79, 123, 135, 174, 185], 40: [11, 36, 54, 113, 156], 59: [4, 19, 52, 64, 108, 112, 183], 62: [44, 85, 101, 175], 71: [71, 130, 138, 172, 191], 82: [28, 45, 78, 111, 178], 84: [49, 103, 133, 151, 186, 199], 90: [5, 29, 62, 90, 120, 147, 152, 190], 92: [0, 59, 67, 110, 139, 194], 93: [65, 121, 125, 188], 115: [18, 82, 131, 160, 162, 171, 189], 117: [107, 116, 117, 159, 165], 124: [30, 39, 99, 104, 144], 125: [23, 86, 93, 184], 126: [1, 10, 50, 56, 57, 126, 181], 143: [25, 51, 68, 80, 114, 143], 149: [13, 27, 42, 48, 98, 102, 149], 150: [32, 77, 146, 176], 152: [34, 43, 55, 66, 94, 142, 161, 163, 164], 153: [12, 16, 41, 153, 168], 163: [9, 35, 72, 81, 87, 95, 157], 168: [17, 26, 70, 73, 91, 109], 172: [7, 60, 75, 76, 154, 166, 169, 196], 186: [33, 84, 132, 193], 189: [92, 97, 134, 137, 167], 197: [127, 173, 179, 197, 198]}, 62, 936.7266623727543))