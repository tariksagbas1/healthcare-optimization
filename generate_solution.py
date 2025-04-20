import ast
from pathlib import Path
from textwrap import shorten

def clean_and_generate_txt(input_path, output_dir="."):
    with open(input_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()[1:] 

    current_entry = []
    entries = []

    
    for line in lines:
        if line.startswith('"sam_global') and current_entry:
            entries.append("".join(current_entry))
            current_entry = [line]
        else:
            current_entry.append(line)
    entries.append("".join(current_entry)) 

    for idx, entry in enumerate(entries):
        try:
            entry = entry.replace('""', '"').replace('\n', ' ').strip()

            
            dataset_index = ''.join(filter(str.isdigit, entry.split(',')[1]))

            
            loc_start = entry.find("[")
            loc_end = entry.find("]", loc_start) + 1
            locations = ast.literal_eval(entry[loc_start:loc_end])

            # Parse Z-value
            zval_frag = entry[loc_end:]
            z_value = float(zval_frag.split(",")[1].strip())

            # Parse Assignments
            assign_start = entry.find("{", loc_end)
            assign_end = entry.find("}", assign_start) + 1
            assignments = ast.literal_eval(entry[assign_start:assign_end])

            # Parse Stage-2
            stage2_start = entry.find("(", assign_end)
            stage2_end = entry.rfind(")") + 1
            stage2_tuple = ast.literal_eval(entry[stage2_start:stage2_end])
            stage2_value = stage2_tuple[0]
            routes = stage2_tuple[1]

            # Format file content
            content = "Stage-1:\n"
            for loc in locations:
                assigned = sorted([x + 1 for x in assignments.get(loc, [])])
                content += f"Healthcenter deployed at {loc + 1}: Communities Assigned = {{{', '.join(map(str, assigned))}}}\n"
            content += f"Objective Value: {z_value:.12f}\n\nStage-2:\n"

            for i, (_, stops) in enumerate(routes.items(), 1):
                converted = ['Depot' if s == 0 else f"Healthcenter at {s + 1}" for s in stops]
                content += f"Route {i}: {' -> '.join(converted)}\n"
            content += f"Objective Value: {stage2_value:.2f}\n"

            # Write to file
            output_path = Path(output_dir) / f"Sol_Instance_{dataset_index}.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

        except Exception as e:
            print(f"Entry {idx} failed: {e} | Snippet: {shorten(entry, width=120)}")

    print("All files generated.")

clean_and_generate_txt("solutions copy.csv", output_dir="./solutions")
