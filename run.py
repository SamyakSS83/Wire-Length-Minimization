import operator
import random
# from visualize_gates import *
import time
import numpy as np
from packer import RectanglePacker, PackedRectangle
# from visualize import Visualize
import subprocess
import sys


def parse_input_file(file_path):
    """
    Reads rectangle dimensions from a file and sorts them by the numeric part of the name.
 
    Args:
        file_path: The path to the input file.
 
    Returns:
        A list of tuples representing rectangle dimensions (width, height) sorted by the name order.
    """
    with open(file_path, "r") as file:
        rectangles = []
        for line in file:
            dimensions = line.split()
            # Extracting index i from gi
            index = int(dimensions[0][1:])
            width, height = int(dimensions[1]), int(dimensions[2])
            rectangles.append((index, (width, height)))
 
    # Sort rectangles by the extracted index
    rectangles.sort(key=lambda x: x[0])
 
    # Return only the sorted dimensions
    return [dim for _, dim in rectangles], [i for i, _ in rectangles]


def do_rectangles_intersect(rect_a, rect_b):
    """
    Checks if two rectangles overlap.

    Args:
        rect_a, rect_b: PackedRectangle objects.

    Returns:
        True if the rectangles overlap, False otherwise.
    """
    return not (
        rect_b.position[0] >= rect_a.position[0] + rect_a.dimensions[0]
        or rect_b.position[0] + rect_b.dimensions[0] <= rect_a.position[0]
        or rect_b.position[1] >= rect_a.position[1] + rect_a.dimensions[1]
        or rect_b.position[1] + rect_b.dimensions[1] <= rect_a.position[1]
    )


def is_solution_valid(packed_rectangles):
    """
    Checks if a packing solution is valid (no overlaps).

    Args:
        packed_rectangles: A list of PackedRectangle objects.

    Returns:
        True if the solution is valid, False otherwise.
    """
    for i, rect_a in enumerate(packed_rectangles):
        for rect_b in packed_rectangles[i + 1 :]:
            if do_rectangles_intersect(rect_a, rect_b):
                return False
    return True


def calculate_packing_efficiency(rectangles, solution_bounds):
    """
    Calculates the packing efficiency of a solution.

    Args:
        rectangles: A list of tuples representing rectangle dimensions.
        solution_bounds: A SolutionBounds object representing the bounding box of the solution.

    Returns:
        The packing efficiency as a percentage.
    """
    total_rectangle_area = np.sum([width * height for width, height in rectangles])
    solution_area = solution_bounds.dimensions[0] * solution_bounds.dimensions[1]
    return 100 * (total_rectangle_area / solution_area)


def determine_bounding_box(packed_rectangles):
    """
    Determines the bounding box (minimum rectangle enclosing all packed rectangles).

    Args:
        packed_rectangles: A list of PackedRectangle objects.

    Returns:
        A SolutionBounds object representing the bounding box.
    """
    max_x, max_y = 0, 0
    for rect in packed_rectangles:
        bottom_right_x = rect.position[0] + rect.dimensions[0]
        bottom_right_y = rect.position[1] + rect.dimensions[1]
        max_x = max(max_x, bottom_right_x)
        max_y = max(max_y, bottom_right_y)
    return SolutionBounds((max_x, max_y))


def generate_naive_solution(rectangles):
    """
    Generates a naive packing solution (placing rectangles side-by-side).

    Args:
        rectangles: A list of tuples representing rectangle dimensions.

    Returns:
        A list of PackedRectangle objects representing the naive solution.
    """
    x, y = 0, 0
    naive_packed_rectangles = []
    for width, height in rectangles:
        position = (x, y)
        naive_packed_rectangles.append(PackedRectangle(position, (width, height)))
        x += width
    return naive_packed_rectangles


def find_optimal_solution(rectangles):
    """
    Finds the optimal packing solution using different sorting strategies.

    Args:
        rectangles: A list of tuples representing rectangle dimensions.

    Returns:
        A list of PackedRectangle objects representing the optimal solution.
    """

    def sort_by_height_then_width(rectangles):
        return sorted(rectangles, key=operator.itemgetter(1, 0), reverse=True)

    def sort_by_width_then_height(rectangles):
        return sorted(rectangles, key=operator.itemgetter(0, 1), reverse=True)

    def sort_by_area(rectangles):
        return sorted(rectangles, key=lambda x: x[0] * x[1], reverse=True)

    def sort_by_max_side(rectangles):
        return sorted(rectangles, key=lambda x: max(x[0], x[1]), reverse=True)
    
    

    sorting_strategies = [
        sort_by_height_then_width,
        sort_by_width_then_height,
        sort_by_area,
        sort_by_max_side,
    ]

    best_solution = None
    min_area = float("inf")

    for strategy in sorting_strategies:
        sorted_rectangles = strategy(rectangles.copy())
        packer = RectanglePacker()
        current_solution = packer.arrange_rectangles(sorted_rectangles)
        solution_bounds = determine_bounding_box(current_solution)
        current_area = solution_bounds.dimensions[0] * solution_bounds.dimensions[1]

        if current_area < min_area:
            min_area = current_area
            best_solution = current_solution

    return best_solution


def save_results_to_file(rectangles, packed_rectangles, output_file_path, bounding_box_width, bounding_box_height, indices):
    """
    Writes the packing solution to a file.

    Args:
        rectangles: A list of tuples representing rectangle dimensions.
        packed_rectangles: A list of PackedRectangle objects.
        output_file_path: The path to the output file.
        bounding_box_width, bounding_box_height: Dimensions of the bounding box.
    """
    rectangles_list = [(rect.dimensions, rect.position) for rect in packed_rectangles]
    # Write to the output file
    with open(output_file_path, "w") as file:
        file.write(f"bounding_box {bounding_box_width} {bounding_box_height}\n")
        for i, (width, height) in zip(indices, rectangles):
            # Find the first occurrence of the rectangle dimensions in the list
            for j, (dimensions, position) in enumerate(rectangles_list):
                if dimensions == (width, height):
                    # Write the position to the file
                    file.write(f"g{i} {position[0]} {position[1]}\n")
                    # Remove the used rectangle from the list
                    rectangles_list.pop(j)
                    break  # Exit the inner loop once the rectangle is found and removed


class SolutionBounds:
    """
    Represents the bounding box of a packing solution.
    """

    def __init__(self, dimensions):
        self.dimensions = dimensions

    def calculate_perimeter(self):
        """
        Calculates the perimeter of the bounding box.

        Returns:
            The perimeter.
        """
        return (self.dimensions[0] + self.dimensions[1]) * 2


import operator
import random

import time
import numpy as np
from packer import RectanglePacker, PackedRectangle  # Assuming you've made the changes to packer.py as well
import os


def Runner(input_file_path, output_file_path):

    rectangles, indices = parse_input_file(input_file_path)

    start_time = time.time()
    optimal_packed_rectangles = find_optimal_solution(rectangles)
    elapsed_time = time.time() - start_time

    print(f"Solution found in {elapsed_time:.2f} seconds")

    optimal_bounds = determine_bounding_box(optimal_packed_rectangles)
    bounding_box_width, bounding_box_height = optimal_bounds.dimensions

    # Write the results to the output file
    save_results_to_file(
        rectangles, optimal_packed_rectangles, output_file_path, bounding_box_width, bounding_box_height, indices
    )

    # try:
    #     import pygame
    #     assert pygame
    #     pass
    # except (ImportError, AssertionError):
    #     print("Pygame not found. Installing...")
    #     subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])

    # visualizer = Visualize(optimal_packed_rectangles, optimal_bounds)
    # visualizer.display()
    
    """
    Implementation of the provided visualizing tool
    """
# Runner("D:/col215/sw2/final/Clusters/cluster0.txt", "D:/col215/sw2/final/Clusters/cluster0_out.txt")