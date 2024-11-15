import random

def generate_test_case(pattern, num_gates=50, total_wires=200):
    gates = []
    wires = []
    pin_counts = {}
    
    for i in range(1, num_gates + 1):
        width = random.randint(2, 25)
        height = random.randint(10, 25)
        num_pins = min(height-1, 20)  # Ensure number of pins doesn't exceed height
        gates.append(f"g{i} {width} {height}")
        pin_counts[f'g{i}'] = num_pins

        pins = []
        for _ in range(num_pins):
            pin_x = random.choice([0, width])  # Pin on left or right side
            pin_y = random.randint(0, height - 1)
            pins.append(f"{pin_x} {pin_y}")
        
        pin_line = " ".join(pins)
        gates.append(f"pins g{i} {pin_line}")
    
    def add_wire():
        gate1, gate2 = random.sample(range(1, num_gates + 1), 2)
        pin1 = random.randint(1, pin_counts[f'g{gate1}'])
        pin2 = random.randint(1, pin_counts[f'g{gate2}'])
        return f"wire g{gate1}.p{pin1} g{gate2}.p{pin2}"

    if pattern == "densely_connected":
        wires = [add_wire() for _ in range(total_wires)]
    
    elif pattern == "sparsely_connected":
        for gate in range(1, num_gates):
            next_gate = gate + 1
            pin1 = random.randint(1, pin_counts[f'g{gate}'])
            pin2 = random.randint(1, pin_counts[f'g{next_gate}'])
            wires.append(f"wire g{gate}.p{pin1} g{next_gate}.p{pin2}")
    
    elif pattern == "multi_clustered":
        clusters = [range(1, 17), range(17, 34), range(34, 51)]
        wires_per_cluster = total_wires // 3
        for cluster in clusters:
            cluster_wires = [add_wire() for _ in range(wires_per_cluster) if random.choice(cluster) and random.choice(cluster)]
            wires.extend(cluster_wires)
    
    elif pattern == "diagonally_sequential":
        for i in range(1, min(num_gates, total_wires)):
            gate1, gate2 = i, i + 1
            pin1 = min(i, pin_counts[f'g{gate1}'])
            pin2 = min(i + 1, pin_counts[f'g{gate2}'])
            wires.append(f"wire g{gate1}.p{pin1} g{gate2}.p{pin2}")
    
    elif pattern == "star_like":
        center = num_gates // 2
        for gate in range(1, num_gates + 1):
            if gate != center and len(wires) < total_wires:
                pin1 = random.randint(1, pin_counts[f'g{center}'])
                pin2 = random.randint(1, pin_counts[f'g{gate}'])
                wires.append(f"wire g{center}.p{pin1} g{gate}.p{pin2}")
    
    elif pattern == "mesh_like":
        for i in range(1, num_gates + 1):
            for j in range(i + 1, min(i + 5, num_gates + 1)):
                if len(wires) < total_wires:
                    pin1 = random.randint(1, pin_counts[f'g{i}'])
                    pin2 = random.randint(1, pin_counts[f'g{j}'])
                    wires.append(f"wire g{i}.p{pin1} g{j}.p{pin2}")
    
    elif pattern == "random":
        wires = [add_wire() for _ in range(total_wires)]

    return gates, wires

def write_to_file(filename, gates, wires):
    with open(filename, "w") as file:
        file.write("\n".join(gates))
        file.write("\n")
        file.write("\n".join(wires))

# Generate test cases for each pattern
patterns = ["densely_connected", "sparsely_connected", "multi_clustered", "diagonally_sequential", "star_like", "mesh_like", "random"]

for pattern in patterns:
    gates, wires = generate_test_case(pattern)
    write_to_file(f"D:/col215/sw2/final/final_submission/Test_Special/{pattern}_input.txt", gates, wires)
    print(f"Generated test case for {pattern} pattern")