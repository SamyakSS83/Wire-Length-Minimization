from collections import deque

class RectanglePacker:
    def __init__(self):
        self.initial_node = None
        self.use_recursion = False
        self.prioritize_tight_fit = True
        # IDEA: keep nodes sorted by smallest, then first found fit will be best.
        # This list grows a lot if max size is increased...  Why?
        self.available_spaces = deque()

    def arrange_rectangles(self, rectangles):
        packed_rectangles = []

        for rect in rectangles:
            if self.initial_node is None:
                self.initial_node = SpaceNode((0, 0), rectangles[0])
                current_space = self.initial_node
                self.available_spaces.append(self.initial_node)
            else:
                current_space = self.locate_space(rect)

            if current_space is not None:
                packed_rectangles.append(self.divide_space(current_space, rect))
            else:
                packed_rectangles.append(self.expand_space(rect))

        return packed_rectangles

    def locate_space(self, dimensions):
        if self.use_recursion:
            return self.recursive_space_search(self.initial_node, dimensions)
        else:
            optimal_space = None

            for space in self.available_spaces:
                if space.can_accommodate(dimensions):
                    if not self.prioritize_tight_fit:
                        return space

                    if optimal_space:
                        if space.is_tighter_fit(optimal_space, dimensions):
                            optimal_space = space
                    else:
                        optimal_space = space

            return optimal_space

    def recursive_space_search(self, current_space, dimensions):
        if current_space:
            if current_space.is_occupied:
                return self.recursive_space_search(current_space.right_child, dimensions) or self.recursive_space_search(current_space.bottom_child, dimensions)
            elif current_space.can_fit(dimensions):
                return current_space

        return None

    def divide_space(self, current_space, dimensions):
        current_space.is_occupied = True
        self.available_spaces.remove(current_space)

        # Check to see if there's space to even split
        if current_space.dimensions[1] - dimensions[1] > 0:
            current_space.bottom_child = SpaceNode((current_space.position[0], current_space.position[1] + dimensions[1]),
                                               (current_space.dimensions[0], current_space.dimensions[1] - dimensions[1]))

            self.available_spaces.appendleft(current_space.bottom_child)

        if current_space.dimensions[0] - dimensions[0] > 0:
            current_space.right_child = SpaceNode((current_space.position[0] + dimensions[0], current_space.position[1]),
                                              (current_space.dimensions[0] - dimensions[0], dimensions[1]))

            self.available_spaces.appendleft(current_space.right_child)

        current_space.dimensions = dimensions
        return PackedRectangle(current_space.position, dimensions)

    def expand_space(self, dimensions):
        # Needs to handle really wide blocks better
        can_expand_down = dimensions[0] <= self.initial_node.dimensions[0]  # check if rectangle width is less than root
        can_expand_right = dimensions[1] <= self.initial_node.dimensions[1]  # check if rectangle height is less than root

        should_expand_down = can_expand_down and (self.initial_node.dimensions[0] >= (self.initial_node.dimensions[1] + dimensions[1]))
        should_expand_right = can_expand_right and (self.initial_node.dimensions[1] >= (self.initial_node.dimensions[0] + dimensions[0]))

        if should_expand_right:
            return self.expand_rightward(dimensions)
        elif should_expand_down:
            return self.expand_downward(dimensions)
        elif can_expand_right:
            return self.expand_rightward(dimensions)
        elif can_expand_down:
            return self.expand_downward(dimensions)

        return None

    def expand_rightward(self, dimensions):
        updated_initial = SpaceNode((0, 0), (self.initial_node.dimensions[0] + dimensions[0], self.initial_node.dimensions[1]))
        updated_initial.is_occupied = True
        updated_initial.bottom_child = self.initial_node
        updated_initial.right_child = SpaceNode((self.initial_node.dimensions[0], 0),
                                            (dimensions[0], self.initial_node.dimensions[1]))

        self.initial_node = updated_initial

        if self.initial_node.right_child.dimensions[0] > 0 and self.initial_node.right_child.dimensions[1] > 0:
            self.available_spaces.appendleft(self.initial_node.right_child)

        return self.divide_space(self.initial_node.right_child, dimensions)

    def expand_downward(self, dimensions):
        updated_initial = SpaceNode((0, 0), (self.initial_node.dimensions[0], self.initial_node.dimensions[1] + dimensions[1]))
        updated_initial.is_occupied = True
        updated_initial.right_child = self.initial_node
        updated_initial.bottom_child = SpaceNode((0, self.initial_node.dimensions[1]),
                                             (self.initial_node.dimensions[0], dimensions[1]))

        self.initial_node = updated_initial

        if self.initial_node.bottom_child.dimensions[0] > 0 and self.initial_node.bottom_child.dimensions[1] > 0:
            self.available_spaces.appendleft(self.initial_node.bottom_child)

        return self.divide_space(self.initial_node.bottom_child, dimensions)


class SpaceNode:
    def __init__(self, position, dimensions):
        self.is_occupied = False
        self.bottom_child = None
        self.right_child = None
        self.position = position
        self.dimensions = dimensions

    def can_fit(self, dimensions):
        return dimensions[1] <= self.dimensions[1] and dimensions[0] <= self.dimensions[0]

    def can_accommodate(self, dimensions):
        return dimensions[1] <= self.dimensions[1] and dimensions[0] <= self.dimensions[0]

    def is_tighter_fit(self, other_node, dimensions):
        return self.dimensions[0] - dimensions[0] < other_node.dimensions[0] - dimensions[0] or self.dimensions[1] - dimensions[1] < other_node.dimensions[1] - dimensions[1]


class PackedRectangle:
    def __init__(self, position, dimensions):
        self.dimensions = dimensions
        self.position = position
        self.rect = position + dimensions