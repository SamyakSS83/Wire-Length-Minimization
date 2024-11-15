import random
import math
from run import *

class defaultdict(dict):
    def __init__(self, default_factory=None):
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            if self.default_factory is None:
                raise
            value = self.default_factory()
            self[key] = value
            return value
        
class CircuitComponent:
    def __init__(self, identifier, width, height, pins):
        self.identifier = identifier
        self.width = width
        self.height = height
        self.pins = pins
        self.x = 0
        self.y = 0

    def reposition(self, new_x, new_y):
        self.x = new_x
        self.y = new_y

    def get_pin_coordinates(self, pin_index):
        pin_x, pin_y = self.pins[pin_index]
        return self.x + pin_x, self.y + pin_y

    def get_footprint(self):
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def is_colliding(self, other):
        x1, y1, x2, y2 = self.get_footprint()
        ox1, oy1, ox2, oy2 = other.get_footprint()
        return not (x1 >= ox2 or x2 <= ox1 or y1 >= oy2 or y2 <= oy1)

class CircuitOptimizer:
    def __init__(self, components, wires, start_temp, cooling_factor):
        self.components = components
        self.wires = wires
        self.temperature = start_temp
        self.cooling_factor = cooling_factor


    def calculate_wire_length(self):
        total = 0
        for (g1, p1, g2, p2) in self.wires:
            x1, y1 = self.components[g1].get_pin_coordinates(p1)
            x2, y2 = self.components[g2].get_pin_coordinates(p2)
            total += abs(x1 - x2) + abs(y1 - y2)
        return total

    def calculate_overlap(self):
        overlap = 0
        components_list = list(self.components.values())
        for i in range(len(components_list)):
            for j in range(i + 1, len(components_list)):
                if components_list[i].is_colliding(components_list[j]):
                    overlap += 1
        return overlap

    def attempt_component_relocation(self):
        component = random.choice(list(self.components.values()))
        max_dim = max(max(c.width, c.height) for c in self.components.values())
        grid_size = max_dim * (len(self.components) + 1)  # Ensure enough space
        new_x = random.randint(0, grid_size - component.width)
        new_y = random.randint(0, grid_size - component.height)
        old_x, old_y = component.x, component.y
        component.reposition(new_x, new_y)
        return component, old_x, old_y
    
    def attempt_component_relocation_small(self):
        component = random.choice(list(self.components.values()))
        old_x, old_y = component.x, component.y
        new_x = old_x + random.randint(1,30)
        new_y = old_y + random.randint(1,30)
        component.reposition(new_x, new_y)
        return component, old_x, old_y
    
    def should_accept_change(self, old_cost, new_cost):
        if new_cost < old_cost:
            return True
        return random.random() < math.exp((old_cost - new_cost) / self.temperature)

    def optimize(self):
        current_wire_length = self.calculate_wire_length()
        current_cost = current_wire_length
        best_cost = current_cost
        best_layout = {c.identifier: (c.x, c.y) for c in self.components.values()}

        x = len(self.components.values())
        if x < 20:
            max_iters = 1000000
        elif x < 40:
            max_iters = 500000
        elif x < 100:
            max_iters = 100000
        elif x < 300:
            max_iters = 50000
        else:
            max_iters = 1000
        for _ in range(max_iters):
            component, old_x, old_y = self.attempt_component_relocation()
            
            new_wire_length = self.calculate_wire_length()
            new_overlap = self.calculate_overlap()
            new_cost = new_wire_length
            
            if self.should_accept_change(current_cost, new_cost) and not new_overlap:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_layout = {c.identifier: (c.x, c.y) for c in self.components.values()}
            else:
                component.reposition(old_x, old_y)
                
            self.temperature *= self.cooling_factor

        return best_layout, best_cost

def initial_compact_placement(components):
    with open(f"temp.txt", 'w') as file:
        for component in components.values():
                file.write(f"{component.identifier} {component.width} {component.height}\n")
    Runner(f"temp.txt", f"temp_out.txt")
        
    with open(f"temp_out.txt", 'r') as file:
        bb = file.readline().split()
        for line in file:
            line = line.strip().split()
            components[line[0]].reposition(int(line[1]), int(line[2]))
    return None
    


def parse_input(input_data):
    lines = input_data.strip().split('\n')
    components = {}
    wires = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if len(line.split()) >= 3 and line.split()[0] != 'pins' and line.split()[0] != 'wire':
            name, width, height = line.split()
            width, height = int(width), int(height)
            i += 1
            pins_line = lines[i].strip().split()[2:]
            pins = [(int(pins_line[j]), int(pins_line[j + 1])) for j in range(0, len(pins_line), 2)]
            components[name] = CircuitComponent(name, width, height, pins)

        elif line.startswith('wire'):
            wire_data = line.split()[1:]
            g1, p1 = wire_data[0].split('.')
            g2, p2 = wire_data[1].split('.')

            p1 = int(p1[1:]) - 1
            p2 = int(p2[1:]) - 1

            if p1 >= len(components[g1].pins):
                raise IndexError(f"Invalid pin index {p1 + 1} for gate {g1}. Gate {g1} only has {len(components[g1].pins)} pins.")
            if p2 >= len(components[g2].pins):
                raise IndexError(f"Invalid pin index {p2 + 1} for gate {g2}. Gate {g2} only has {len(components[g2].pins)} pins.")

            wires.append((g1, p1, g2, p2))

        i += 1

    return components, wires


def output_result(components, wire_length):
    bounding_box = calculate_bounding_box(components)
    
    print(f"bounding_box {bounding_box[0]} {bounding_box[1]} {bounding_box[2]} {bounding_box[3]}")
    print(f"wire_length {wire_length}")
    
    for component in components.values():
        print(f"{component.identifier} {component.x} {component.y}")

def calculate_bounding_box(components):
    min_x = min(c.x for c in components.values())
    min_y = min(c.y for c in components.values())
    max_x = max(c.x + c.width for c in components.values())
    max_y = max(c.y + c.height for c in components.values())
    return (max_x, max_y)


def find_clusters(components, wires):
    # Create an adjacency list of the components based on the wires
    adj_list = defaultdict(set)
    for g1, _, g2, _ in wires:
        adj_list[g1].add(g2)
        adj_list[g2].add(g1)

    visited = set()
    clusters = []

    def dfs(component, cluster):
        visited.add(component)
        cluster.add(component)
        for neighbor in adj_list[component]:
            if neighbor not in visited:
                dfs(neighbor, cluster)

    for component in components.keys():
        if component not in visited:
            current_cluster = set()
            dfs(component, current_cluster)
            clusters.append(current_cluster)
    # print(clusters)
    return clusters

def read_input_file(filename):
    with open(filename, 'r') as file:
        return file.read()

def write_output_file(filename, components, wire_length):
    with open(filename, 'w') as file:
        bounding_box = calculate_bounding_box(components)
        
        file.write(f"bounding_box {bounding_box[0]} {bounding_box[1]}\n")
        
        for component in components.values():
            file.write(f"{component.identifier} {component.x} {component.y}\n")
            
        file.write(f"wire_length {wire_length}\n")

def place_clusters_in_line(components, clusters, wires, cooling_factor=0.99, start_temp=1000):
    x_offset = 0
    total_wire_length = 0

    num_clusters = len(clusters)
    print(f"Total number of clusters: {num_clusters}")

    for i, cluster in enumerate(clusters, 1):
        print(f"\nStarting cluster {i}/{num_clusters}")
        cluster_components = {comp: components[comp] for comp in cluster}
        cluster_wires = [wire for wire in wires if wire[0] in cluster and wire[2] in cluster]

        # Optimize the current cluster layout
        initial_compact_placement(cluster_components)
        optimizer = CircuitOptimizer(cluster_components, cluster_wires, start_temp, cooling_factor)
        best_layout, cluster_wire_length = optimizer.optimize()
        total_wire_length += cluster_wire_length

        # Update component positions with the current x_offset and place them in the final layout
        x_min = min(x for identifier, (x, _) in best_layout.items())
        cluster_width = max(components[identifier].width + x - x_min for identifier, (x, _) in best_layout.items())
        y_min = min(y for identifier, (_, y) in best_layout.items())
        for identifier, (x, y) in best_layout.items():
            components[identifier].reposition(x + x_offset - x_min, y - y_min)

        # Calculate the width of the current cluster and update x_offset for the next cluster
        x_offset += cluster_width  

    return total_wire_length


# Function to parse the input data from input.txt
def input_parse(data):
    gates = {}
    pins = {}
    wires = []

    for line in data:
        tokens = line.split()

        if tokens[0].startswith('g'):
            # Parsing gate dimensions
            gate_name = tokens[0]
            width, height = int(tokens[1]), int(tokens[2])
            gates[gate_name] = {"width": width, "height": height}
        
        elif tokens[0] == "pins":
            # Parsing pin coordinates
            gate_name = tokens[1]
            pin_coords = [(int(tokens[i]), int(tokens[i+1])) for i in range(2, len(tokens), 2)]
            pins[gate_name] = pin_coords

        elif tokens[0] == "wire":
            # Parsing wire connections
            wire_from = tokens[1].split('.')
            wire_to = tokens[2].split('.')
            wires.append((wire_from, wire_to))
    
    return gates, pins, wires

# Function to parse the gate positions from output.txt
def parse_gate_positions(data):
    gate_positions = {}
    
    for line in data:
        tokens = line.split()

        if tokens[0].startswith('g'):
            gate_name = tokens[0]
            x, y = int(tokens[1]), int(tokens[2])
            gate_positions[gate_name] = (x, y)
    
    return gate_positions

# Function to calculate pin coordinates based on gate placement
def calculate_pin_coordinates(gate_positions, gates, pins):
    pin_positions = {}

    for gate, position in gate_positions.items():
        gate_x, gate_y = position
        pin_positions[gate] = [(gate_x + px, gate_y + py) for (px, py) in pins[gate]]

    return pin_positions

# Function to calculate Manhattan distance between two points
def calculate_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# Function to compute a 2D matrix of distances between all pairs of pins
def calculate_all_pin_distances(pin_positions):
    all_pins = []  # To store all pins and their coordinates
    pin_names = []  # To keep track of which pin belongs to which gate and index
    gate_names = []  # To track which gate each pin belongs to

    # Flattening the pin_positions dictionary into a list of pins with their coordinates and names
    for gate, pin_list in pin_positions.items():
        for i, pin_coords in enumerate(pin_list):
            pin_name = f"{gate}.p{i+1}"
            all_pins.append(pin_coords)
            pin_names.append(pin_name)
            gate_names.append(gate)  # Keep track of which gate each pin belongs to
    
    # Create a 2D matrix for distances
    distance_matrix = [[0] * len(all_pins) for _ in range(len(all_pins))]

    # Calculate distances between all pairs of pins
    for i in range(len(all_pins)):
        for j in range(len(all_pins)):
            if i != j:
                if gate_names[i] == gate_names[j]:
                    # Pins belong to the same gate, set distance to infinity
                    distance_matrix[i][j] = math.inf
                else:
                    # Calculate Manhattan distance for pins from different gates
                    distance_matrix[i][j] = calculate_distance(all_pins[i], all_pins[j])
    
    return distance_matrix, pin_names, gate_names

# Function to compute a 2D matrix of True/False for connected pins or same gate pins
def calculate_connection_matrix(pin_positions, pin_names, gate_names, wires):
    connection_matrix = [[False] * len(pin_names) for _ in range(len(pin_names))]

    # Explicitly mark False for pins belonging to the same gate
    for i in range(len(pin_names)):
        for j in range(len(pin_names)):
            if gate_names[i] == gate_names[j]:
                connection_matrix[i][j] = False  # Pins of the same gate must have False

    # Mark True for connected pins based on the wire connections
    for wire in wires:
        gate1, pin1 = wire[0]
        gate2, pin2 = wire[1]
        pin1_idx = int(pin1[1:]) - 1
        pin2_idx = int(pin2[1:]) - 1
        pin1_full = f"{gate1}.p{pin1_idx + 1}"
        pin2_full = f"{gate2}.p{pin2_idx + 1}"

        if pin1_full in pin_names and pin2_full in pin_names:
            idx1 = pin_names.index(pin1_full)
            idx2 = pin_names.index(pin2_full)
            connection_matrix[idx1][idx2] = True
            
    
    return connection_matrix

# New function to return a sequential list of pin coordinates
def get_pin_coordinates_in_order(pin_positions):
    ordered_pin_coordinates = []
    for gate, pin_list in pin_positions.items():
        ordered_pin_coordinates.extend(pin_list)  # Flatten the coordinates into a single list
    return ordered_pin_coordinates

def Wire_Len(output_filename, input_filename):
    
    
    with open(input_filename, 'r') as f:
        input_data = f.readlines()
    # Reading gate positions from the output file
    with open(output_filename, 'r') as f:
        output_data = f.readlines()
    # Parse the input and output data
    gates, pins, wires = input_parse(input_data)
    gate_positions = parse_gate_positions(output_data)
    # Calculate pin coordinates based on gate positions
    pin_positions = calculate_pin_coordinates(gate_positions, gates, pins)
    # Calculate distances between all pairs of pins
    distance_matrix, pin_names, gate_names = calculate_all_pin_distances(pin_positions)
    # Calculate connection matrix (True/False for connected pins or same gate pins)
    connection_matrix = calculate_connection_matrix(pin_positions, pin_names, gate_names, wires)
    # Get the list of pin coordinates in sequential order
    ordered_pin_coordinates = get_pin_coordinates_in_order(pin_positions)
    total_wire_length=0
    i=0
    j=0
    for i in range(0,len(connection_matrix)):
        temp=connection_matrix[i]

        if True in temp:
            bounding_x=list()
            bounding_y=list()
            bounding_x.append(ordered_pin_coordinates[i][0])
            bounding_y.append(ordered_pin_coordinates[i][1])



            for j in range(0,len(temp)):
                if connection_matrix[i][j]:
                    bounding_x.append(ordered_pin_coordinates[j][0])
                    bounding_y.append(ordered_pin_coordinates[j][1])
            
            xmin=min(bounding_x)
            xmax=max(bounding_x)
            ymin=min(bounding_y)
            ymax=max(bounding_y)

            total_wire_length=total_wire_length+ xmax-xmin+ymax-ymin
            
    return total_wire_length

def Solver(input_filename, output_filename, cooling_factor, start_temp):
    input_data = read_input_file(input_filename)

    components, wires = parse_input(input_data)
    clusters = find_clusters(components, wires)
    _ = place_clusters_in_line(components, clusters, wires, cooling_factor, start_temp)
    write_output_file(output_filename, components, 10)
    total_wire_length = Wire_Len(output_filename, input_filename)
    write_output_file(output_filename, components, total_wire_length)

    print(f"Output has been written to {output_filename}")

import time
start = time.time()   
input_filename = "input.txt"
output_filename = "output.txt"
Solver(input_filename,
    output_filename ,
    cooling_factor = 0.95,
    start_temp = 1000)
end = time.time()
print(end - start)
