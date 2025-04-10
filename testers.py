from models_gurobi import g_model_apx1, g_model_apx2, g_model_new_obj, g_model_iterative, g_model_global, ifs1

def test1(test_model, dataset_index_, n_p_, m_p_):


    with open(f"./datasets/Instance_{dataset_index_}.txt", "r") as file: # Read from file
            lines = file.readlines()
    p = []
    for line in lines[2:(n_p_+2)]:  # skip first two lines
            values = list(line.split()) 
            population = int(values[4])
            p.append(population)  # add populations for each community i
    total_population = sum(p)
    C_p_ = total_population / m_p_
    
    if C_p_ % 1 != 0:
        return "Undefined"
    else:
        return test_model(dataset_index = dataset_index_, n_p = n_p_, C_p = C_p_, m_p = m_p_)

def test2(test_model, dataset_index):
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()
        n = int(lines[0].split()[0])
        m = int(lines[0].split()[1])
        C = int(lines[2].split()[3])
    
    return test_model(dataset_index = dataset_index, n_p = n, C_p = C, m_p = m) 

# For ifs testing
def test3(test_model, dataset_index, units:list):
    with open(f"./datasets/Instance_{dataset_index}.txt", "r") as file: # Read from file
        lines = file.readlines()
        n = int(lines[0].split()[0])
        m = int(lines[0].split()[1])
        C = int(lines[2].split()[3])

    return test_model(dataset_index = dataset_index, n_p = n, C_p = C, m_p = m, units = units)

def test_list1(test_model, dataset_index, n_list, m):
    lister = []
    for x in n_list:
        lister.append(test1(test_model = test_model, dataset_index_ = dataset_index, n_p_ = x, m_p_ = m))
    for x in range(len(lister)):
        print(f" for n = {n_list[x]} : {lister[x]}")

print(test3(ifs1, 1, [12, 13, 27, 41, 48, 50, 53, 61, 75, 78]))