#!/usr/bin/env python3
"""
generate_solution_files.py

Reads new_solutions.csv and, for each row, writes
solutions_new/Sol_Instance_<dataset>.txt in the required template.
"""

import re, ast, os

CSV_PATH  = "new_solutions.csv"
OUT_DIR   = "solutions_new"        # <-- all files land here

_END = re.compile(
    r'^(?P<prefix>.*?),\s*'                   # everything before last 2 fields
    r'(?P<locs>"\[[^\]]*\]"),\s*'             # "[ ... ]"
    r'(?P<asgn>"\{.*\}")\s*$'                 # "{ ... }"
)

def generate_solution_files(csv_path=CSV_PATH, output_dir=OUT_DIR):
    # make the folder if it doesn’t exist
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # skip header
    for lineno, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line:
            continue

        m = _END.match(line)
        if not m:
            print(f"Line {lineno}: cannot parse tail → skipping")
            continue

        prefix, loc_q, asg_q = m.group("prefix"), m.group("locs"), m.group("asgn")

        fields = [p.strip() for p in prefix.split(",")]
        if len(fields) != 10:
            print(f"Line {lineno}: expected 10 prefix fields, got {len(fields)} → skipping")
            continue

        (_model, dataset, Zs, _time,
         maxWs, minWs, alphas,
         maxDs, minDs, betas) = fields

        try:
            Z     = float(Zs)
            max_w = float(maxWs);  min_w = float(minWs)
            alpha = float(alphas)
            max_d = float(maxDs);  min_d = float(minDs)
            beta  = float(betas)
        except ValueError as e:
            print(f"Line {lineno}: numeric parse error ({e}) → skipping")
            continue

        try:
            locations   = ast.literal_eval(loc_q[1:-1])     # strip quotes
            assignments = ast.literal_eval(asg_q[1:-1])     # strip quotes
        except Exception as e:
            print(f"Line {lineno}: literal_eval error ({e}) → skipping")
            continue
        if not isinstance(assignments, dict):
            print(f"Line {lineno}: assignments not a dict → skipping")
            continue

        # Build file content
        out_lines = []
        for unit in sorted(assignments, key=int):
            comms_str = ", ".join(str(c) for c in sorted(assignments[unit]))
            out_lines.append(
                f"Healthcenter deployed at {unit}: Communities Assigned = {{{comms_str}}}"
            )
        out_lines.append("")
        out_lines.append(f"Objective Value: {Z}")
        out_lines.append("")
        gap_w = max_w - min_w
        out_lines += [
            "Workload Fairness Check:",
            f"  Min workload = {min_w:.2f}, Max workload = {max_w:.2f}",
            f"  Workload Gap = {gap_w:.2f} (Threshold α = {alpha})",
            ""
        ]
        gap_d = max_d - min_d
        out_lines += [
            "Distance Fairness Check:",
            f"  Min Distance = {min_d:.2f}, Max Distance = {max_d:.2f}",
            f"  Distance Gap = {gap_d:.2f} (Threshold β = {beta})"
        ]

        # Write file
        out_path = os.path.join(output_dir, f"Sol_Instance_{dataset}.txt")
        with open(out_path, "w", encoding="utf-8") as fout:
            fout.write("\n".join(out_lines))
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    generate_solution_files()
